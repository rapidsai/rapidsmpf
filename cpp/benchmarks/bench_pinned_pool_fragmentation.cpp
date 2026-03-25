/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Benchmark: impact of memory fragmentation on PinnedMemoryResource
 * =================================================================
 *
 * Compares a variable-size pinned memory pool (cuda::pinned_memory_pool) against
 * fixed-block pools (cucascade::fixed_size_host_memory_resource) with 1 MiB, 4 MiB,
 * and 8 MiB block sizes by measuring the largest single allocation achievable after
 * intentional fragmentation.
 *
 * Each benchmark iteration runs three phases:
 *
 *   Phase 1 — Fill
 *     Allocate random-sized buffers drawn uniformly from [1 MiB, max_fill_MiB] (a
 *     benchmark argument) until the pool is exhausted (OOM).  The same RNG seed is used
 *     for all modes so the allocation pattern is identical.
 *
 *   Phase 2 — Fragment
 *     Randomly free individual allocations (uniform index sampling; already-freed slots
 *     are skipped) until the cumulative freed bytes reach kPoolFreeFactor × kMaxPool.
 *     This leaves the pool with ~50 % free memory scattered across non-contiguous holes.
 *
 *   Phase 3 — Probe max allocatable size
 *     Attempt a single allocation starting at 1 MiB, doubling the size each step up to
 *     the free-target, then bisect (1 MiB granularity) between the last success and the
 *     first failure to find the exact largest allocatable size.
 *
 * Reported counters:
 *   max_alloc_GiB      — largest single allocation that succeeded in the fragmented pool
 *   free_target_GiB    — bytes freed before probing (kPoolFreeFactor × kMaxPool)
 *   block_size_MiB     — fixed block size in MiB (0 = variable-size pool)
 *   max_fill_MiB       — upper bound of the random fill-request distribution (MiB)
 *   pool_free_factor   — fraction of kMaxPool freed before probing
 *
 * Benchmark arguments: {block_size_MiB, max_fill_MiB}
 *   block_size_MiB ∈ {0, 1, 4, 8}   (0 → variable-size pool)
 *   max_fill_MiB   ∈ {128, 256, 512, 1024}
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <ranges>
#include <stdexcept>
#include <utility>
#include <vector>

#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>

#include <cuda/memory_resource>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/pinned_memory_resource.hpp>
#include <rapidsmpf/system_info.hpp>

namespace {

constexpr std::uint64_t kRngSeed = 42;
constexpr std::size_t kInitialPool = 8ULL * 1024 * 1024 * 1024;  // 8 GiB
constexpr std::size_t kMaxPool = 16ULL * 1024 * 1024 * 1024;  // 16 GiB
constexpr std::size_t kMinFillBytes = 1ULL << 20;  // 1 MiB
constexpr double kPoolFreeFactor = 0.50;
constexpr std::size_t kProbeStep = 1ULL << 20;  // 1 MiB bisection granularity

rapidsmpf::PinnedPoolProperties make_pool_properties() {
    return {
        .initial_pool_size = kInitialPool,
        .max_pool_size = std::optional<std::size_t>{kMaxPool},
    };
}

/// Find the largest allocatable size in [0, upper_bound] using doubling then bisection
/// (kProbeStep granularity). @p can_alloc(n) attempts one allocation of @p n bytes and
/// returns true on success.
template <typename CanAllocFn>
[[nodiscard]] std::size_t probe_max_alloc(CanAllocFn can_alloc, std::size_t upper_bound) {
    // Recursive doubling to find a loose upper bound.
    std::size_t lo = 0;
    std::size_t probe = kProbeStep;
    while (probe <= upper_bound) {
        if (!can_alloc(probe))
            break;
        lo = probe;
        if (probe >= upper_bound)
            break;
        probe = std::min(probe * 2, upper_bound);
    }
    // lo = last success (0 if even kProbeStep failed), probe = first failure.
    std::size_t hi = std::min(probe, upper_bound);

    // Bisection with kProbeStep granularity.
    while (lo + kProbeStep <= hi) {
        std::size_t const mid = ((lo + (hi - lo) / 2) / kProbeStep) * kProbeStep;
        if (mid <= lo)
            break;
        if (can_alloc(mid)) {
            lo = mid;
        } else {
            hi = mid - kProbeStep;
        }
    }
    return lo;
}

// ─── Variable-size pool ───────────────────────────────────────────────────────

struct VarAlloc {
    void* ptr;
    std::size_t size;
};

/// Phase 1 (variable): fill pool with random-sized allocations until OOM.
[[nodiscard]] std::vector<VarAlloc> var_fill(
    rapidsmpf::PinnedMemoryResource& mr,
    rmm::cuda_stream_view stream,
    std::mt19937_64& rng,
    std::size_t max_fill_bytes
) {
    std::uniform_int_distribution<std::size_t> dist(kMinFillBytes, max_fill_bytes);
    std::vector<VarAlloc> live;

    while (true) {
        std::size_t const req = dist(rng);
        void* p = nullptr;
        try {
            p = mr.allocate(stream, req);
            stream.synchronize();
        } catch (std::bad_alloc const&) {
            break;
        } catch (cuda::cuda_error const&) {
            break;
        } catch (rapidsmpf::cuda_error const&) {
            break;
        }
        live.push_back({p, req});
    }
    return live;
}

/// Phase 2 (variable): randomly free live allocations until freed >= free_target.
/// Picks random indices; skips already-freed slots (ptr == nullptr).
void var_fragment(
    rapidsmpf::PinnedMemoryResource& mr,
    rmm::cuda_stream_view stream,
    std::vector<VarAlloc>& live,
    std::mt19937_64& rng,
    std::size_t free_target
) {
    std::uniform_int_distribution<std::size_t> idx_dist(0, live.size() - 1);
    std::size_t freed = 0;
    while (freed < free_target) {
        std::size_t const idx = idx_dist(rng);
        if (!live[idx].ptr)
            continue;
        mr.deallocate(stream, live[idx].ptr, live[idx].size);
        freed += live[idx].size;
        live[idx].ptr = nullptr;
    }
    stream.synchronize();

    auto [first, last] =
        std::ranges::remove_if(live, [](VarAlloc const& a) { return !a.ptr; });
    live.erase(first, last);
}

/// Phase 3 (variable): probe for the largest single allocation in the fragmented pool.
[[nodiscard]] std::size_t var_probe_max(
    rapidsmpf::PinnedMemoryResource& mr,
    rmm::cuda_stream_view stream,
    std::size_t upper_bound
) {
    return probe_max_alloc(
        [&](std::size_t size) -> bool {
            try {
                void* p = mr.allocate(stream, size);
                if (p) {
                    mr.deallocate(stream, p, size);
                }
                stream.synchronize();
                return true;
            } catch (std::bad_alloc const&) {
                return false;
            } catch (cuda::cuda_error const&) {
                return false;
            } catch (rapidsmpf::cuda_error const&) {
                return false;
            }
        },
        upper_bound
    );
}

// ─── Fixed-block pool ─────────────────────────────────────────────────────────

using FixedAlloc = rapidsmpf::PinnedMemoryResource::FixedSizedBlocksAllocation;

/// Phase 1 (fixed): fill pool with random-sized allocations until OOM.
[[nodiscard]] std::vector<FixedAlloc> fixed_fill(
    rapidsmpf::PinnedMemoryResource& mr, std::mt19937_64& rng, std::size_t max_fill_bytes
) {
    std::uniform_int_distribution<std::size_t> dist(kMinFillBytes, max_fill_bytes);
    std::vector<FixedAlloc> live;

    while (true) {
        std::size_t const req = dist(rng);
        try {
            live.push_back(mr.allocate_fixed_sized(req));
        } catch (std::bad_alloc const&) {
            break;
        } catch (cuda::cuda_error const&) {
            break;
        } catch (rapidsmpf::cuda_error const&) {
            break;
        }
    }
    return live;
}

/// Phase 2 (fixed): randomly free live allocations until freed >= free_target.
/// Picks random indices; skips already-freed slots (null unique_ptr).
/// RAII `FixedSizedBlocksAllocation` returns blocks to the pool on reset().
void fixed_fragment(
    std::vector<FixedAlloc>& live, std::mt19937_64& rng, std::size_t free_target
) {
    std::uniform_int_distribution<std::size_t> idx_dist(0, live.size() - 1);
    std::size_t freed = 0;
    while (freed < free_target) {
        std::size_t const idx = idx_dist(rng);
        if (!live[idx])
            continue;
        freed += live[idx]->size_bytes();
        live[idx].reset();  // RAII: blocks returned to pool
    }

    // Compact: remove reset (null) entries.
    auto [first, last] =
        std::ranges::remove_if(live, [](FixedAlloc const& a) { return !a; });
    live.erase(first, last);
}

/// Phase 3 (fixed): probe for the largest single allocation in the fragmented pool.
[[nodiscard]] std::size_t fixed_probe_max(
    rapidsmpf::PinnedMemoryResource& mr, std::size_t upper_bound
) {
    return probe_max_alloc(
        [&](std::size_t size) -> bool {
            try {
                std::ignore =
                    mr.allocate_fixed_sized(size);  // RAII release on scope exit
                return true;
            } catch (std::bad_alloc const&) {
                return false;
            } catch (cuda::cuda_error const&) {
                return false;
            } catch (rapidsmpf::cuda_error const&) {
                return false;
            }
        },
        upper_bound
    );
}

// ─────────────────────────────────────────────────────────────────────────────

/// @p block_size == 0  → variable-size pool
/// @p block_size  > 0  → fixed-block pool with that block size
void BM_PinnedPoolFragmentedMaxAlloc(benchmark::State& state) {
    if (!rapidsmpf::is_pinned_memory_resources_supported()) {
        state.SkipWithMessage("pinned memory not supported on system");
        return;
    }

    RAPIDSMPF_CUDA_TRY(cudaFree(nullptr));

    auto const block_size = static_cast<std::size_t>(state.range(0)) << 20;
    auto const max_fill_bytes = static_cast<std::size_t>(state.range(1)) << 20;
    rmm::cuda_stream stream{rmm::cuda_stream::flags::non_blocking};
    auto const props = make_pool_properties();
    auto const free_target =
        static_cast<std::size_t>(kPoolFreeFactor * static_cast<double>(kMaxPool));

    for (auto _ : state) {
        state.PauseTiming();

        std::mt19937_64 rng{kRngSeed};
        std::size_t max_allocatable = 0;

        if (block_size == 0) {
            rapidsmpf::PinnedMemoryResource mr{rapidsmpf::get_current_numa_node(), props};

            auto live = var_fill(mr, stream.view(), rng, max_fill_bytes);
            var_fragment(mr, stream.view(), live, rng, free_target);

            max_allocatable = var_probe_max(mr, stream.view(), free_target);

            std::ranges::for_each(live, [&](auto const& a) {
                mr.deallocate(stream.view(), a.ptr, a.size);
            });
            stream.view().synchronize();
        } else {
            auto mr = rapidsmpf::PinnedMemoryResource::make_fixed_sized_if_available(
                rapidsmpf::get_current_numa_node(), props, block_size
            );
            if (!mr) {
                state.SkipWithMessage("fixed-size pinned resource unavailable");
                return;
            }
            auto live = fixed_fill(*mr, rng, max_fill_bytes);
            fixed_fragment(live, rng, free_target);

            max_allocatable = fixed_probe_max(*mr, free_target);
            live.clear();  // RAII dealloc
        }

        state.ResumeTiming();
        benchmark::DoNotOptimize(max_allocatable);

        state.counters["free_target_GiB"] =
            static_cast<double>(free_target) / static_cast<double>(1ULL << 30);
        state.counters["max_alloc_GiB"] =
            static_cast<double>(max_allocatable) / static_cast<double>(1ULL << 30);
        state.counters["block_size_MiB"] =
            static_cast<double>(block_size) / static_cast<double>(1ULL << 20);
        state.counters["pool_free_factor"] = static_cast<double>(kPoolFreeFactor);
        state.counters["max_fill_MiB"] =
            static_cast<double>(max_fill_bytes) / static_cast<double>(1ULL << 20);
    }
}

void register_fragmentation_args(benchmark::internal::Benchmark* b) {
    for (int64_t const max_fill_mib : {128, 256, 512, 1024}) {
        b->Args({0, max_fill_mib});  // variable-size pool
        b->Args({1, max_fill_mib});  // fixed 1 MiB blocks
        b->Args({4, max_fill_mib});  // fixed 4 MiB blocks
        b->Args({8, max_fill_mib});  // fixed 8 MiB blocks
    }
}

}  // namespace

BENCHMARK(BM_PinnedPoolFragmentedMaxAlloc)
    ->Apply(register_fragmentation_args)
    ->Iterations(1)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
