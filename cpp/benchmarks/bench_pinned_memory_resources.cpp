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
#include <cuda_runtime.h>

#include <rmm/cuda_stream.hpp>

#include <rapidsmpf/buffer/pinned_memory_resource.hpp>
#include <rapidsmpf/error.hpp>

namespace {

// inspired by
// https://github.com/rapidsai/rmm/blob/branch-25.12/cpp/benchmarks/async_priming/async_priming_bench.cpp
// benchmark

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
    if (!rapidsmpf::is_pinned_memory_resources_supported()) {
        state.SkipWithMessage(
            "PinnedMemoryResource is not supported for CUDA versions "
            "below " RAPIDSMPF_PINNED_MEM_RES_MIN_CUDA_VERSION_STR
        );
        return;
    }

    constexpr int num_allocations = 100;

    auto [pool, mr] = factory(initial_pool_size);

    std::vector<void*> allocations;
    allocations.reserve(num_allocations);

    rmm::cuda_stream stream{rmm::cuda_stream::flags::non_blocking};

    for (auto _ : state) {
        auto t0 = std::chrono::high_resolution_clock::now();

        // First allocation - measure latency to this specific call
        allocations.push_back(mr->allocate(stream, allocation_size));
        stream.synchronize();
        auto t1 = std::chrono::high_resolution_clock::now();

        for (int i = 1; i < num_allocations; ++i) {
            allocations.push_back(mr->allocate(stream, allocation_size));
        }

        stream.synchronize();
        auto t2 = std::chrono::high_resolution_clock::now();

        for (auto* ptr : allocations) {
            mr->deallocate(stream, ptr, allocation_size);
        }
        allocations.clear();
        stream.synchronize();

        auto t3 = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < num_allocations; ++i) {
            allocations.push_back(mr->allocate(stream, allocation_size));
        }

        stream.synchronize();
        auto t4 = std::chrono::high_resolution_clock::now();

        // Calculate metrics
        auto latency_to_first =
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        auto first_round_duration_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t0).count();
        auto second_round_duration_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(t4 - t3).count();

        auto first_round_throughput =
            (static_cast<double>(num_allocations * allocation_size) * 1e9)
            / first_round_duration_ns;
        auto second_round_throughput =
            (static_cast<double>(num_allocations * allocation_size) * 1e9)
            / second_round_duration_ns;

        state.counters["latency_to_first_ns"] = latency_to_first;
        state.counters["first_round_throughput"] = first_round_throughput;
        state.counters["second_round_throughput"] = second_round_throughput;

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
void BM_AsyncConstructionTime(
    benchmark::State& state, MRFactoryFunc factory, size_t initial_pool_size
) {
    if (!rapidsmpf::is_pinned_memory_resources_supported()) {
        state.SkipWithMessage(
            "PinnedMemoryResource is not supported for CUDA versions "
            "below " RAPIDSMPF_PINNED_MEM_RES_MIN_CUDA_VERSION_STR
        );
        return;
    }

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

/**
 * @brief Benchmark to compare device to host copy vs device to stream-ordered pinned copy
 */
template <typename MRFactoryFunc>
void BM_DeviceToHostCopyComparison(
    benchmark::State& state, MRFactoryFunc factory, size_t copy_size, size_t n_copies
) {
    if (!rapidsmpf::is_pinned_memory_resources_supported()) {
        state.SkipWithMessage(
            "PinnedMemoryResource is not supported for CUDA versions "
            "below " RAPIDSMPF_PINNED_MEM_RES_MIN_CUDA_VERSION_STR
        );
        return;
    }

    auto [pool, mr] = factory(0);

    rmm::cuda_stream stream{rmm::cuda_stream::flags::non_blocking};

    rmm::device_buffer device_buf(copy_size, stream);
    stream.synchronize();

    std::vector<std::vector<std::byte>> host_bufs;
    host_bufs.reserve(n_copies);
    for (size_t i = 0; i < n_copies; ++i) {
        host_bufs.emplace_back(copy_size);
    }

    std::vector<rapidsmpf::PinnedHostBuffer> pinned_bufs;
    pinned_bufs.reserve(n_copies);
    for (size_t i = 0; i < n_copies; ++i) {
        pinned_bufs.emplace_back(copy_size, stream, mr);
    }
    stream.synchronize();

    for (auto _ : state) {
        state.PauseTiming();

        stream.synchronize();

        state.ResumeTiming();

        auto t0 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < n_copies; ++i) {
            RAPIDSMPF_CUDA_TRY(cudaMemcpy(
                host_bufs[i].data(), device_buf.data(), copy_size, cudaMemcpyDefault
            ));
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < n_copies; ++i) {
            RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
                pinned_bufs[i].data(),
                device_buf.data(),
                copy_size,
                cudaMemcpyDefault,
                stream.value()
            ));
        }
        stream.synchronize();
        auto t2 = std::chrono::high_resolution_clock::now();

        benchmark::DoNotOptimize(device_buf);
        benchmark::DoNotOptimize(host_bufs);
        benchmark::DoNotOptimize(pinned_bufs);

        auto host_copy_time_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        auto pinned_copy_time_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();


        // Calculate bandwidth (GB/s)
        auto total_bytes = static_cast<double>(copy_size * n_copies);
        auto host_bandwidth_gbps = total_bytes / host_copy_time_ns;
        auto pinned_bandwidth_gbps = total_bytes / pinned_copy_time_ns;

        state.counters["host_copy_time_ns"] = host_copy_time_ns;
        state.counters["pinned_copy_time_ns"] = pinned_copy_time_ns;
        state.counters["host_bandwidth_gbps"] = host_bandwidth_gbps;
        state.counters["pinned_bandwidth_gbps"] = pinned_bandwidth_gbps;
        state.counters["speedup"] = static_cast<double>(host_copy_time_ns)
                                    / static_cast<double>(pinned_copy_time_ns);
    }
}

}  // namespace

// Register benchmarks

// 10MB allocations with no priming
BENCHMARK_CAPTURE(BM_AsyncPrimingImpact, unprimed, &make_pinned_resource, 0, 10 << 20)
    ->Unit(benchmark::kMicrosecond);

// 10MB allocations with 1000MB priming
BENCHMARK_CAPTURE(
    BM_AsyncPrimingImpact, primed, &make_pinned_resource, 1000 << 20, 10 << 20
)
    ->Unit(benchmark::kMicrosecond);

// no priming
BENCHMARK_CAPTURE(BM_AsyncConstructionTime, unprimed, &make_pinned_resource, 0)
    ->Unit(benchmark::kMicrosecond);
// 1GB priming
BENCHMARK_CAPTURE(BM_AsyncConstructionTime, primed_1GB, &make_pinned_resource, 1 << 30)
    ->Unit(benchmark::kMicrosecond);
// 4GB priming
BENCHMARK_CAPTURE(BM_AsyncConstructionTime, primed_4GB, &make_pinned_resource, 4 << 30)
    ->Unit(benchmark::kMicrosecond);

static auto register_device_to_host_copy_benchmarks = [] {
    struct BenchConfig {
        size_t copy_size_mb;
        size_t n_copies;
    };

    constexpr std::array<BenchConfig, 12> configs = {{
        {.copy_size_mb = 1, .n_copies = 1},
        {.copy_size_mb = 1, .n_copies = 10},
        {.copy_size_mb = 1, .n_copies = 100},
        {.copy_size_mb = 4, .n_copies = 1},
        {.copy_size_mb = 4, .n_copies = 10},
        {.copy_size_mb = 4, .n_copies = 100},
        {.copy_size_mb = 10, .n_copies = 1},
        {.copy_size_mb = 10, .n_copies = 10},
        {.copy_size_mb = 10, .n_copies = 100},
        {.copy_size_mb = 100, .n_copies = 1},
        {.copy_size_mb = 100, .n_copies = 10},
        {.copy_size_mb = 100, .n_copies = 100},
    }};

    for (const auto& config : configs) {
        std::string name = "BM_DeviceToHostCopyComparison/"
                           + std::to_string(config.copy_size_mb) + "MB_"
                           + std::to_string(config.n_copies) + "copies";

        benchmark::RegisterBenchmark(
            name.c_str(),
            [](benchmark::State& state, auto factory, size_t copy_size, size_t n_copies) {
                BM_DeviceToHostCopyComparison(state, factory, copy_size, n_copies);
            },
            &make_pinned_resource,
            config.copy_size_mb << 20,
            config.n_copies
        )
            ->Unit(benchmark::kMicrosecond);
    }

    return 0;
}();

BENCHMARK_MAIN();
