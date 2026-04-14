/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Benchmark: CUDA driver pinned memory pool fragmentation
 * ========================================================
 *
 * Standalone benchmark (no rapidsmpf dependency) that measures the largest
 * single allocation achievable in a CUDA driver pinned memory pool
 * (cudaMemPool_t) after intentional fragmentation.
 *
 * Only the driver pool (cudaMemPool_t with cudaMemAllocationTypePinned) is
 * benchmarked.  The pool is created fresh per iteration, pre-warmed to
 * kInitialPool bytes, and never releases memory to the OS between phases.
 *
 * Scenario: 1 CUDA stream, 25 % free factor, fill sizes 128 / 256 / 512 MiB.
 *
 * Benchmark arguments: {max_fill_MiB, free_pct, num_producer_threads}
 *   max_fill_MiB         ∈ {128, 256, 512}
 *   free_pct             = 25   (fraction of kMaxPool freed before probing)
 *   num_producer_threads ∈ {1, 2, 4}
 *
 * Three phases per iteration:
 *
 *   Phase 1 — Fill
 *     @p num_producer_threads concurrent threads allocate random-sized buffers
 *     drawn uniformly from [1 MiB, max_fill_MiB] on a shared single CUDA
 *     stream until the pool returns cudaErrorMemoryAllocation.  The same RNG
 *     seed base is used across runs for reproducibility.
 *
 *   Phase 2 — Fragment
 *     Threads randomly free live allocations (skipping already-freed slots)
 *     until cumulative freed bytes reach free_factor × kMaxPool.  This leaves
 *     ~25 % of the pool free but scattered across non-contiguous holes.
 *
 *   Phase 3 — Probe max allocatable size
 *     Doubling then bisection at 1 MiB granularity finds the largest single
 *     allocation that succeeds in the fragmented pool.
 *
 * Reported counters:
 *   max_alloc_GiB        — largest single allocation that succeeded
 *   free_target_GiB      — bytes freed before probing (free_factor × kMaxPool)
 *   max_fill_MiB         — upper bound of the fill-request distribution (MiB)
 *   pool_free_factor     — fraction of kMaxPool freed before probing
 *   num_producer_threads — concurrent threads used during fill and fragment
 */

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <random>
#include <ranges>
#include <stdexcept>
#include <string>
#include <vector>

#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>

#include <rmm/aligned.hpp>

#include <cuda/memory_resource> 

namespace {

// ─── CUDA error checking ──────────────────────────────────────────────────────

#define CUDA_CHECK(expr)                                                           \
    do {                                                                           \
        cudaError_t _err = (expr);                                                 \
        if (_err != cudaSuccess) {                                                 \
            throw std::runtime_error(                                              \
                std::string("CUDA error in " __FILE__ ":") +                      \
                std::to_string(__LINE__) + " — " + cudaGetErrorString(_err)        \
            );                                                                     \
        }                                                                          \
    } while (0)

// ─── CUDA event RAII wrapper ──────────────────────────────────────────────────

/// Lightweight RAII wrapper around cudaEvent_t.
/// Uses cudaEventDisableTiming so events have minimal overhead.
struct CudaEvent {
    cudaEvent_t event = nullptr;

    CudaEvent() { CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming)); }
    ~CudaEvent() noexcept {
        if (event) {
            cudaEventDestroy(event);
        }
    }

    CudaEvent(CudaEvent const&)            = delete;
    CudaEvent& operator=(CudaEvent const&) = delete;
    CudaEvent(CudaEvent&& o) noexcept : event{o.event} { o.event = nullptr; }

    void record(cudaStream_t stream) { CUDA_CHECK(cudaEventRecord(event, stream)); }

    /// Make the given stream wait for this event before executing further work.
    void stream_wait(cudaStream_t stream) const {
        CUDA_CHECK(cudaStreamWaitEvent(stream, event, 0 /*flags*/));
    }

    /// Create, record, and return a shared CudaEvent on @p stream.
    static std::shared_ptr<CudaEvent> make_shared_record(cudaStream_t stream) {
        auto e = std::make_shared<CudaEvent>();
        e->record(stream);
        return e;
    }
};

// ─── Pool type alias ─────────────────────────────────────────────────────────

/// cuda::mr::shared_resource<cuda::pinned_memory_pool> owns a reference-counted
/// cuda::pinned_memory_pool (backed by cudaMemPool_t, cudaMemAllocationTypePinned).
using PinnedPool = cuda::mr::shared_resource<cuda::pinned_memory_pool>;

// ─── Constants ────────────────────────────────────────────────────────────────

constexpr std::uint64_t kRngSeed      = 42;
constexpr std::size_t   kInitialPool  = 8ULL  * 1024 * 1024 * 1024;  // 8 GiB
constexpr std::size_t   kMaxPool      = 16ULL * 1024 * 1024 * 1024;  // 16 GiB
constexpr std::size_t   kMinFillBytes = 1ULL << 20;                   // 1 MiB
constexpr std::size_t   kProbeStep    = 1ULL << 20;                   // 1 MiB

// ─── Phase implementations ────────────────────────────────────────────────────

struct VarAlloc {
    void*                      ptr   = nullptr;
    std::size_t                size  = 0;
    std::shared_ptr<CudaEvent> event;
};

/// Phase 1: fill the pool with random-sized allocations until OOM.
///
/// @p num_threads producer threads run concurrently; each has its own RNG
/// seeded from kRngSeed + thread_id.  All threads allocate on the shared
/// @p stream.  cudaMallocFromPoolAsync is thread-safe for concurrent calls to
/// the same pool on the same stream.  A shared OOM flag stops all threads as
/// soon as any one hits an allocation failure.
[[nodiscard]] std::vector<VarAlloc> var_fill(
    PinnedPool&  pool,
    cudaStream_t stream,
    std::size_t  max_fill_bytes,
    std::size_t  num_threads
) {
    std::mutex            mtx;
    std::vector<VarAlloc> live;
    std::atomic<bool>     oom{false};

    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (std::size_t t = 0; t < num_threads; ++t) {
        futures.push_back(std::async(std::launch::async, [&, t]() {
            std::mt19937_64                             rng{kRngSeed + t};
            std::uniform_int_distribution<std::size_t> dist{kMinFillBytes, max_fill_bytes};

            while (!oom.load(std::memory_order_relaxed)) {
                std::size_t const req = dist(rng);
                void*             ptr = nullptr;
                try {
                    ptr = pool.allocate(cuda::stream_ref{stream}, req, rmm::CUDA_ALLOCATION_ALIGNMENT);
                } catch (cuda::cuda_error const&) {
                    oom.store(true, std::memory_order_relaxed);
                    break;
                }
                // Schedule dummy work so the stream is genuinely busy when events
                // are recorded; pattern is derived from the pointer to vary writes.
                auto const pattern = static_cast<int>(reinterpret_cast<uintptr_t>(ptr) & 0xFF);
                cudaMemsetAsync(ptr, pattern, req, stream);
                // Record an event so Phase 2 can safely order its deallocations
                // after this allocation has been enqueued on the stream.
                auto ev = CudaEvent::make_shared_record(stream);
                std::lock_guard lock{mtx};
                live.emplace_back(ptr, req, std::move(ev));
            }
        }));
    }
    for (auto& f : futures) {
        f.get();
    }
    return live;
}

/// Phase 2: randomly free live allocations until freed bytes >= free_target.
///
/// @p num_threads threads run concurrently.  A mutex protects slot selection,
/// the freed counter, and slot nulling so no allocation is freed twice.
/// Each deallocation is stream-ordered after the corresponding allocation's
/// event, preserving CUDA stream semantics.
void var_fragment(
    PinnedPool&           pool,
    cudaStream_t          stream,
    std::vector<VarAlloc>& live,
    std::size_t           free_target,
    std::size_t           num_threads
) {
    std::mutex  mtx;
    std::size_t freed = 0;

    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (std::size_t t = 0; t < num_threads; ++t) {
        futures.push_back(std::async(std::launch::async, [&, t]() {
            // Offset seeds from var_fill threads for an independent sequence.
            std::mt19937_64                             rng{kRngSeed + 1000 + t};
            std::uniform_int_distribution<std::size_t> idx_dist{0, live.size() - 1};

            while (true) {
                void*                      ptr  = nullptr;
                std::size_t                size = 0;
                std::shared_ptr<CudaEvent> ev;
                {
                    std::lock_guard lock{mtx};
                    if (freed >= free_target) {
                        break;
                    }
                    std::size_t idx = idx_dist(rng);
                    while (!live[idx].ptr) {
                        idx = idx_dist(rng);
                    }
                    ptr           = live[idx].ptr;
                    size          = live[idx].size;
                    ev            = std::move(live[idx].event);
                    live[idx].ptr = nullptr;
                    freed += size;
                }
                ev->stream_wait(stream);
                pool.deallocate(cuda::stream_ref{stream}, ptr, size, rmm::CUDA_ALLOCATION_ALIGNMENT);
            }
        }));
    }
    for (auto& f : futures) {
        f.get();
    }

    // Compact: remove freed (null ptr) entries.
    auto [first, last] =
        std::ranges::remove_if(live, [](VarAlloc const& a) { return !a.ptr; });
    live.erase(first, last);
}

/// Phase 3: probe for the largest single allocation in the fragmented pool.
/// Uses doubling then bisection at kProbeStep granularity to find the largest
/// size in [0, upper_bound] for which a single allocation succeeds.
[[nodiscard]] std::size_t var_probe_max(
    PinnedPool& pool, cudaStream_t stream, std::size_t upper_bound
) {
    auto can_alloc = [&](std::size_t size) -> bool {
        try {
            void* p = pool.allocate(cuda::stream_ref{stream}, size, rmm::CUDA_ALLOCATION_ALIGNMENT);
            pool.deallocate(cuda::stream_ref{stream}, p, size, rmm::CUDA_ALLOCATION_ALIGNMENT);
            return true;
        } catch (cuda::cuda_error const&) {
            return false;
        }
    };

    // Doubling phase: find a loose upper bound.
    std::size_t lo    = 0;
    std::size_t probe = kProbeStep;
    while (probe <= upper_bound) {
        if (!can_alloc(probe)) {
            break;
        }
        lo = probe;
        if (probe >= upper_bound) {
            break;
        }
        probe = std::min(probe * 2, upper_bound);
    }
    // lo = last success (0 if even kProbeStep failed), probe = first failure.
    std::size_t hi = std::min(probe, upper_bound);

    // Bisection with kProbeStep granularity.
    while (lo + kProbeStep <= hi) {
        std::size_t const mid = ((lo + (hi - lo) / 2) / kProbeStep) * kProbeStep;
        if (mid <= lo) {
            break;
        }
        if (can_alloc(mid)) {
            lo = mid;
        } else {
            hi = mid - kProbeStep;
        }
    }
    return lo;
}

// ─── Benchmark function ───────────────────────────────────────────────────────

/// Benchmark arguments: {max_fill_MiB, free_pct, num_producer_threads}
void BM_DriverPinnedPoolFragmentation(benchmark::State& state) {
    // Initialise the CUDA context before timing.
    CUDA_CHECK(cudaFree(nullptr));

    auto const max_fill_bytes       = static_cast<std::size_t>(state.range(0)) << 20;
    auto const free_factor          = static_cast<double>(state.range(1)) / 100.0;
    auto const num_producer_threads = static_cast<std::size_t>(state.range(2));
    auto const free_target =
        static_cast<std::size_t>(free_factor * static_cast<double>(kMaxPool));

    // A single non-blocking stream is shared across all phases and threads.
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    for (auto _ : state) {
        state.PauseTiming();

        // Fresh pool per iteration; pre-warm cost is excluded from timing.
        // cuda::memory_pool_properties sets release_threshold to max by default
        // (pool never returns pages to the OS) and warms up initial_pool_size bytes
        // via an internal alloc+free on a private stream at construction.
        auto pool = cuda::mr::make_shared_resource<cuda::pinned_memory_pool>(
            0,  // NUMA node 0
            cuda::memory_pool_properties{
                .initial_pool_size = kInitialPool,
                .max_pool_size     = kMaxPool,
            }
        );

        auto live = var_fill(pool, stream, max_fill_bytes, num_producer_threads);
        var_fragment(pool, stream, live, free_target, num_producer_threads);

        std::size_t max_allocatable = var_probe_max(pool, stream, free_target);

        // Drain remaining live allocations before destroying the pool.
        for (auto const& a : live) {
            a.event->stream_wait(stream);
            pool.deallocate(cuda::stream_ref{stream}, a.ptr, a.size, rmm::CUDA_ALLOCATION_ALIGNMENT);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        state.ResumeTiming();
        benchmark::DoNotOptimize(max_allocatable);

        state.counters["free_target_GiB"] =
            static_cast<double>(free_target) / static_cast<double>(1ULL << 30);
        state.counters["max_alloc_GiB"] =
            static_cast<double>(max_allocatable) / static_cast<double>(1ULL << 30);
        state.counters["pool_free_factor"]      = free_factor;
        state.counters["max_fill_MiB"] =
            static_cast<double>(max_fill_bytes) / static_cast<double>(1ULL << 20);
        state.counters["num_producer_threads"] =
            static_cast<double>(num_producer_threads);
        state.SetLabel("driver pool");
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
}

void register_args(benchmark::Benchmark* b) {
    for (int64_t const max_fill_mib : {128, 256, 512}) {
        for (int64_t const free_pct : {25}) {
            for (int64_t const num_threads : {1, 2, 4}) {
                b->Args({max_fill_mib, free_pct, num_threads});
            }
        }
    }
}

}  // namespace

BENCHMARK(BM_DriverPinnedPoolFragmentation)
    ->Apply(register_args)
    ->Iterations(1)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
