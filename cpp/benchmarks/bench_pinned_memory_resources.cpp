/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <chrono>
#include <cstddef>
#include <iostream>
#include <memory>
#include <vector>

#include <benchmark/benchmark.h>

#include <rmm/cuda_stream.hpp>

#include <rapidsmpf/buffer/pinned_memory_resource.hpp>

namespace {

/**
 * @brief Factory function to create a cuda_async_memory_resource with priming
 */
inline auto make_pinned_resource(size_t initial_pool_size) {
    auto pool = std::make_unique<rapidsmpf::PinnedMemoryPool>(
        0, rapidsmpf::PinnedPoolProperties{.initial_pool_size = initial_pool_size}
    );
    auto resource = std::make_shared<rapidsmpf::PinnedMemoryResource>(*pool);
    return std::make_pair(std::move(pool), std::move(resource));
}

/**
 * @brief Benchmark to measure the impact of async allocator priming
 */
template <typename MRFactoryFunc>
void BM_AsyncPrimingImpact(
    benchmark::State& state,
    MRFactoryFunc factory,
    size_t initial_pool_size,
    size_t allocation_size
) {
    constexpr int num_allocations = 100;

    // Create memory resource
    auto [pool, mr] = factory(initial_pool_size);

    // Storage for allocations
    std::vector<void*> allocations;
    allocations.reserve(num_allocations);

    rmm::cuda_stream stream{rmm::cuda_stream::flags::non_blocking};

    for (auto _ : state) {
        // Measure latency to first allocation
        auto start_time = std::chrono::high_resolution_clock::now();

        // First allocation - measure latency to this specific call
        allocations.push_back(mr->allocate(stream, allocation_size));
        stream.synchronize();
        auto first_allocation_time = std::chrono::high_resolution_clock::now();

        // Continue with remaining allocations in first round
        for (int i = 1; i < num_allocations; ++i) {
            allocations.push_back(mr->allocate(stream, allocation_size));
        }

        stream.synchronize();
        auto first_round_end = std::chrono::high_resolution_clock::now();

        // Deallocate all
        for (auto* ptr : allocations) {
            mr->deallocate(stream, ptr, allocation_size);
        }
        allocations.clear();

        // Second round of allocations
        for (int i = 0; i < num_allocations; ++i) {
            allocations.push_back(mr->allocate(stream, allocation_size));
        }

        stream.synchronize();
        auto second_round_end = std::chrono::high_resolution_clock::now();

        // Calculate metrics
        auto latency_to_first = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    first_allocation_time - start_time
        )
                                    .count();
        auto first_round_duration_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                first_round_end - start_time
            )
                .count();
        auto second_round_duration_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                second_round_end - first_round_end
            )
                .count();

        // Calculate throughput (bytes per second)
        auto first_round_throughput =
            (static_cast<double>(num_allocations * allocation_size) * 1e9)
            / first_round_duration_ns;
        auto second_round_throughput =
            (static_cast<double>(num_allocations * allocation_size) * 1e9)
            / second_round_duration_ns;

        // Set benchmark counters
        state.counters["latency_to_first_ns"] = latency_to_first;
        state.counters["first_round_throughput"] = first_round_throughput;
        state.counters["second_round_throughput"] = second_round_throughput;

        // Clean up for next iteration
        for (auto* ptr : allocations) {
            mr->deallocate(stream, ptr, allocation_size);
        }
        allocations.clear();
        stream.synchronize();
    }
}

/**
 * @brief Benchmark to measure construction time with and without priming
 */
template <typename MRFactoryFunc>
void BM_AsyncConstructionTime(benchmark::State& state, MRFactoryFunc factory, size_t initial_pool_size) {
    for (auto _ : state) {
        auto start_time = std::chrono::high_resolution_clock::now();
        auto [pool, mr] = factory(initial_pool_size);
        benchmark::DoNotOptimize(pool);
        benchmark::DoNotOptimize(mr);
        auto end_time = std::chrono::high_resolution_clock::now();

        auto construction_time =
            std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time)
                .count();

        state.counters["construction_time_us"] = construction_time;
    }
}

}  // namespace

// Register benchmarks
BENCHMARK_CAPTURE(
    BM_AsyncPrimingImpact, unprimed, &make_pinned_resource, 0, 10 << 20
)  // 1MB allocations
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_CAPTURE(
    BM_AsyncPrimingImpact, primed, &make_pinned_resource, 1000 << 20, 10 << 20
)  // 1MB allocations with 100MB priming
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_CAPTURE(BM_AsyncConstructionTime, unprimed, &make_pinned_resource, 0)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_CAPTURE(BM_AsyncConstructionTime, primed, &make_pinned_resource, 1000 << 20)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();