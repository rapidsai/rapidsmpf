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
 *   block_size_MiB     — fixed block size in MiB (0 = variable-size pool modes)
 *   block_tag          — raw first benchmark argument (INT_MAX / INT_MAX-1 / 1 / 4 / 8)
 *   max_fill_MiB       — upper bound of the random fill-request distribution (MiB)
 *   pool_free_factor   — fraction of kMaxPool freed before probing
 *
 * Benchmark arguments: {block_tag, max_fill_MiB, free_pct, num_streams,
 * num_producer_threads} block_tag ∈ {INT_MAX, INT_MAX-1, 1, 4, 8} INT_MAX     →
 * variable-size rapidsmpf::PinnedMemoryResource (cuda pinned pool) INT_MAX - 1 →
 * variable-size rmm::pool_memory_resource over pinned_host_memory_resource 1, 4, 8     →
 * fixed-block rapidsmpf pool (block size in MiB) max_fill_MiB          ∈ {128, 256, 512,
 * 1024} free_pct              ∈ {25, 50}   (percentage of kMaxPool to free before
 * probing) num_streams           ∈ {1, 4, 8}  (stream pool size; always 1 for fixed-block
 * pools) num_producer_threads  ∈ {1, 2, 4}  (concurrent threads used during fill and
 * fragment phases; always 1 for fixed-block pools)
 */

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
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
#include <rmm/cuda_stream_pool.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <rapidsmpf/cuda_event.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/pinned_memory_resource.hpp>
#include <rapidsmpf/system_info.hpp>

namespace {

/// Schedule dummy work on allocated pinned memory to make streams actually busy
/// Uses cudaMemsetAsync to create GPU work without requiring CUDA kernels
void schedule_dummy_work(void* ptr, std::size_t size, rmm::cuda_stream_view stream) {
    if (size == 0)
        return;

    // Use cudaMemsetAsync to create GPU work on the pinned memory
    // This creates real GPU work that will be synchronized by events/stream sync
    auto const pattern = static_cast<int>(reinterpret_cast<uintptr_t>(ptr) & 0xFF);
    RAPIDSMPF_CUDA_TRY(cudaMemsetAsync(ptr, pattern, size, stream.value()));
}

/// First benchmark range dimension: variable rapidsmpf pinned pool (distinct from fixed
/// MiB sizes).
constexpr std::int64_t kBlockTagRapidsmpfVariablePool =
    static_cast<std::int64_t>(INT_MAX);
/// First benchmark range dimension: RMM coalescing pool over pinned host upstream.
constexpr std::int64_t kBlockTagRmmPinnedPool = static_cast<std::int64_t>(INT_MAX) - 1;

constexpr std::uint64_t kRngSeed = 42;
constexpr std::size_t kInitialPool = 8ULL * 1024 * 1024 * 1024;  // 8 GiB
constexpr std::size_t kMaxPool = 16ULL * 1024 * 1024 * 1024;  // 16 GiB
constexpr std::size_t kMinFillBytes = 1ULL << 20;  // 1 MiB
constexpr std::size_t kProbeStep = 1ULL << 20;  // 1 MiB bisection granularity

std::string get_block_tag_name(std::int64_t block_tag) {
    switch (block_tag) {
    case kBlockTagRapidsmpfVariablePool:
        return "driver pool";
    case kBlockTagRmmPinnedPool:
        return "rmm pool";
    default:
        return "fs pool " + std::to_string(block_tag) + "MB";
    }
}

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

void sync_streams(rmm::cuda_stream_pool& stream_pool) {
    for (std::size_t i = 0; i < stream_pool.get_pool_size(); ++i) {
        stream_pool.get_stream(i).synchronize();
    }
}

// ─── Variable-size pool (rmm::device_async_resource_ref) ────────────────────

struct VarAlloc {
    void* ptr;
    std::size_t size;
    std::shared_ptr<rapidsmpf::CudaEvent> event;
};

/// Phase 1 (variable): fill pool with random-sized allocations until OOM.
/// @p num_threads producer threads run concurrently, each with its own RNG (seeded from
/// @p kRngSeed + thread_id). All threads push into a shared mutex-protected @p live
/// vector. A shared OOM flag stops all threads as soon as any one hits an allocation
/// failure. Streams are drawn round-robin from @p stream_pool; all streams are
/// synchronised before returning.
[[nodiscard]] std::vector<VarAlloc> var_fill(
    rmm::device_async_resource_ref mr,
    rmm::cuda_stream_pool& stream_pool,
    std::size_t max_fill_bytes,
    std::size_t num_threads
) {
    std::mutex mtx;
    std::vector<VarAlloc> live;
    std::atomic<bool> oom{false};

    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (std::size_t t = 0; t < num_threads; ++t) {
        futures.push_back(std::async(std::launch::async, [&, t]() {
            std::mt19937_64 rng{kRngSeed + t};
            std::uniform_int_distribution<std::size_t> dist(
                kMinFillBytes, max_fill_bytes
            );
            while (!oom.load(std::memory_order_relaxed)) {
                std::size_t const req = dist(rng);
                void* p = nullptr;
                auto alloc_stream = stream_pool.get_stream();
                try {
                    p = mr.allocate(alloc_stream, req);
                } catch (std::bad_alloc const&) {
                    oom.store(true, std::memory_order_relaxed);
                    break;
                } catch (cuda::cuda_error const&) {
                    oom.store(true, std::memory_order_relaxed);
                    break;
                } catch (rapidsmpf::cuda_error const&) {
                    oom.store(true, std::memory_order_relaxed);
                    break;
                }
                // Schedule some dummy work to make the stream busy
                schedule_dummy_work(p, req, alloc_stream);
                // Record event on the allocating stream
                auto event = rapidsmpf::CudaEvent::make_shared_record(alloc_stream);
                std::lock_guard lock{mtx};
                live.push_back({p, req, std::move(event)});
            }
        }));
    }

    for (auto& f : futures) {
        f.get();
    }

    return live;
}

/// Phase 2 (variable): randomly free live allocations until freed >= free_target.
/// @p num_threads threads run concurrently. A mutex protects index selection, the freed
/// counter, and slot nulling so threads never double-free the same slot. Streams are
/// drawn round-robin from @p stream_pool; all streams are synchronised before compacting
/// the live list.
void var_fragment(
    rmm::device_async_resource_ref mr,
    rmm::cuda_stream_pool& stream_pool,
    std::vector<VarAlloc>& live,
    std::size_t free_target,
    std::size_t num_threads
) {
    std::mutex mtx;
    std::size_t freed = 0;

    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (std::size_t t = 0; t < num_threads; ++t) {
        futures.push_back(std::async(std::launch::async, [&, t]() {
            // Offset seeds from var_fill threads to produce an independent sequence.
            std::mt19937_64 rng{kRngSeed + 1000 + t};
            std::uniform_int_distribution<std::size_t> idx_dist(0, live.size() - 1);
            while (true) {
                std::size_t idx;
                void* ptr = nullptr;
                std::size_t size = 0;
                {
                    std::lock_guard lock{mtx};
                    if (freed >= free_target)
                        break;
                    idx = idx_dist(rng);
                    while (!live[idx].ptr) {
                        idx = idx_dist(rng);
                    }
                    ptr = live[idx].ptr;
                    size = live[idx].size;
                    live[idx].ptr = nullptr;
                    freed += size;
                }
                auto dealloc_stream = stream_pool.get_stream();
                // Wait for allocation to complete before deallocating
                live[idx].event->stream_wait(dealloc_stream);
                mr.deallocate(dealloc_stream, ptr, size);
            }
        }));
    }

    for (auto& f : futures) {
        f.get();
    }

    auto [first, last] =
        std::ranges::remove_if(live, [](VarAlloc const& a) { return !a.ptr; });
    live.erase(first, last);
}

/// Phase 3 (variable): probe for the largest single allocation in the fragmented pool.
[[nodiscard]] std::size_t var_probe_max(
    rmm::device_async_resource_ref mr,
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

/// @p block_tag is kBlockTagRapidsmpfVariablePool or kBlockTagRmmPinnedPool →
/// variable-size pool; otherwise MiB count for fixed-block rapidsmpf pool (1, 4, 8).
void BM_PinnedPoolFragmentedMaxAlloc(benchmark::State& state) {
    if (!rapidsmpf::is_pinned_memory_resources_supported()) {
        state.SkipWithMessage("pinned memory not supported on system");
        return;
    }

    RAPIDSMPF_CUDA_TRY(cudaFree(nullptr));

    std::int64_t const block_tag = state.range(0);
    bool const use_rapidsmpf_variable = (block_tag == kBlockTagRapidsmpfVariablePool);
    bool const use_rmm_variable = (block_tag == kBlockTagRmmPinnedPool);
    bool const use_variable_pool = use_rapidsmpf_variable || use_rmm_variable;

    std::size_t const block_size_bytes =
        use_variable_pool ? 0U : (static_cast<std::size_t>(block_tag) << 20);

    auto const max_fill_bytes = static_cast<std::size_t>(state.range(1)) << 20;
    auto const free_factor = static_cast<double>(state.range(2)) / 100.0;
    auto const num_streams = static_cast<std::size_t>(state.range(3));
    auto const num_producer_threads = static_cast<std::size_t>(state.range(4));
    rmm::cuda_stream_pool stream_pool{num_streams};
    auto const props = make_pool_properties();
    auto const free_target =
        static_cast<std::size_t>(free_factor * static_cast<double>(kMaxPool));

    for (auto _ : state) {
        state.PauseTiming();

        std::size_t max_allocatable = 0;

        if (use_rapidsmpf_variable) {
            rapidsmpf::PinnedMemoryResource mr{rapidsmpf::get_current_numa_node(), props};
            rmm::device_async_resource_ref mr_ref{mr};

            auto live =
                var_fill(mr_ref, stream_pool, max_fill_bytes, num_producer_threads);
            var_fragment(mr_ref, stream_pool, live, free_target, num_producer_threads);

            auto probe_stream = stream_pool.get_stream();
            max_allocatable = var_probe_max(mr_ref, probe_stream, free_target);

            std::ranges::for_each(live, [&](auto const& a) {
                a.event->stream_wait(probe_stream);
                mr.deallocate(probe_stream, a.ptr, a.size);
            });

            sync_streams(stream_pool);
        } else if (use_rmm_variable) {
            rmm::mr::pinned_host_memory_resource pinned_upstream{};
            rmm::mr::pool_memory_resource<rmm::mr::pinned_host_memory_resource> pool_mr{
                pinned_upstream, kInitialPool, std::optional<std::size_t>{kMaxPool}
            };
            rmm::device_async_resource_ref pool_ref{pool_mr};

            auto live =
                var_fill(pool_ref, stream_pool, max_fill_bytes, num_producer_threads);
            var_fragment(pool_ref, stream_pool, live, free_target, num_producer_threads);

            auto probe_stream = stream_pool.get_stream();
            max_allocatable = var_probe_max(pool_ref, probe_stream, free_target);

            std::ranges::for_each(live, [&](auto const& a) {
                a.event->stream_wait(probe_stream);
                pool_mr.deallocate(probe_stream, a.ptr, a.size);
            });

            sync_streams(stream_pool);
        } else {
            auto mr = rapidsmpf::PinnedMemoryResource::make_fixed_sized_if_available(
                rapidsmpf::get_current_numa_node(), props, block_size_bytes
            );
            if (!mr) {
                state.SkipWithMessage("fixed-size pinned resource unavailable");
                return;
            }
            std::mt19937_64 rng{kRngSeed};
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
            static_cast<double>(block_size_bytes) / static_cast<double>(1ULL << 20);
        state.counters["pool_free_factor"] = free_factor;
        state.counters["max_fill_MiB"] =
            static_cast<double>(max_fill_bytes) / static_cast<double>(1ULL << 20);
        state.counters["num_streams"] = static_cast<double>(num_streams);
        state.counters["num_producer_threads"] =
            static_cast<double>(num_producer_threads);
        state.SetLabel(get_block_tag_name(block_tag));
    }
}

void register_fragmentation_args(benchmark::Benchmark* b) {
    for (int64_t const free_pct : {25 /* , 50 */}) {
        for (int64_t const max_fill_mib : {64, 128, 256, 512 /* , 1024 */}) {
            // Variable pools: sweep stream pool size and producer thread count.
            for (int64_t const num_streams : {1, 4, 8}) {
                for (int64_t const num_threads : {1, 2, 4}) {
                    b->Args(
                        {kBlockTagRapidsmpfVariablePool,
                         max_fill_mib,
                         free_pct,
                         num_streams,
                         num_threads}
                    );
                    b->Args(
                        {kBlockTagRmmPinnedPool,
                         max_fill_mib,
                         free_pct,
                         num_streams,
                         num_threads}
                    );
                }
            }
            // Fixed-block pools are stream-agnostic and single-threaded.
            b->Args({1, max_fill_mib, free_pct, 1, 1});  // fixed 1 MiB blocks
            // b->Args({4, max_fill_mib, free_pct, 1, 1});  // fixed 4 MiB blocks
            // b->Args({8, max_fill_mib, free_pct, 1, 1});  // fixed 8 MiB blocks
        }
    }
}

}  // namespace

BENCHMARK(BM_PinnedPoolFragmentedMaxAlloc)
    ->Apply(register_fragmentation_args)
    ->Iterations(1)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
