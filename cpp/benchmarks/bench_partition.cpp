/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>
#include <vector>

#include <benchmark/benchmark.h>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <rapidsmpf/cudf/partition.hpp>

// Helper function to create a table with a single int column
std::unique_ptr<cudf::table> create_int_table(
    cudf::size_type num_rows, rmm::cuda_stream_view stream
) {
    auto data = rmm::device_buffer(size_t(num_rows) * sizeof(int32_t), stream);
    auto validity = rmm::device_buffer(0, stream);  // No nulls

    auto column = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT32},
        num_rows,
        std::move(data),
        std::move(validity),
        0
    );

    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(std::move(column));
    return std::make_unique<cudf::table>(std::move(columns));
}

static void BM_PartitionAndPack(benchmark::State& state) {
    const int64_t local_size = int64_t(state.range(1)) * 1000000;
    int num_rows = int(local_size / int64_t(sizeof(int32_t)));

    const int num_partitions = state.range(1);

    rmm::cuda_stream_view stream = rmm::cuda_stream_default;

    // Create a CUDA memory resource
    auto cuda_mr = std::make_unique<rmm::mr::cuda_memory_resource>();

    // Get total GPU memory
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    size_t total_memory = prop.totalGlobalMem;

    // Calculate 50% of GPU memory
    auto pool_size = static_cast<size_t>(total_memory * 0.5);

    // Create a pool memory resource with 50% of GPU memory
    auto pool_mr =
        std::make_unique<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>(
            cuda_mr.get(), pool_size
        );

    // Create input table
    auto table = create_int_table(num_rows, stream);

    // Columns to hash (just the first column)
    std::vector<cudf::size_type> columns_to_hash{0};

    for (auto _ : state) {
        auto pack_partitions = rapidsmpf::partition_and_pack(
            *table,
            columns_to_hash,
            num_partitions,
            cudf::hash_id::HASH_MURMUR3,
            cudf::DEFAULT_HASH_SEED,
            stream,
            *pool_mr
        );
        benchmark::DoNotOptimize(pack_partitions);
        cudaStreamSynchronize(stream);
    }

    // Set metrics
    state.SetBytesProcessed(state.iterations() * local_size);
    state.counters["num_rows"] = num_rows;
    state.counters["total_nparts"] = num_partitions;
    state.counters["splits"] = num_partitions;
}

static void BM_PartitionAndPackCurrentImpl(benchmark::State& state) {
    const int nranks = state.range(0);
    const int64_t local_size = int64_t(state.range(1)) * 1000000;
    const int num_partitions = state.range(2);

    int total_npartitions = nranks * num_partitions;
    int num_rows = int(local_size / int64_t(sizeof(int32_t)) / int64_t(num_partitions));

    rmm::cuda_stream_view stream = rmm::cuda_stream_default;

    // Create a CUDA memory resource
    auto cuda_mr = std::make_unique<rmm::mr::cuda_memory_resource>();

    // Get total GPU memory
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    size_t total_memory = prop.totalGlobalMem;

    // Calculate 50% of GPU memory
    auto pool_size = static_cast<size_t>(total_memory * 0.5);

    // Create a pool memory resource with 50% of GPU memory
    auto pool_mr =
        std::make_unique<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>(
            cuda_mr.get(), pool_size
        );

    // Create input table
    auto table = create_int_table(num_rows, stream);

    // Columns to hash (just the first column)
    std::vector<cudf::size_type> columns_to_hash{0};

    for (auto _ : state) {
        for (int i = 0; i < num_partitions; i++) {
            auto pack_partitions = rapidsmpf::partition_and_pack(
                *table,
                columns_to_hash,
                total_npartitions,
                cudf::hash_id::HASH_MURMUR3,
                cudf::DEFAULT_HASH_SEED,
                stream,
                *pool_mr
            );
            benchmark::DoNotOptimize(pack_partitions);
        }
        cudaStreamSynchronize(stream);
    }

    // Set metrics
    state.SetBytesProcessed(state.iterations() * local_size);
    state.counters["num_rows"] = num_rows;
    state.counters["total_nparts"] = total_npartitions;
    state.counters["splits"] = total_npartitions * num_partitions;
}

// Custom argument generator for the benchmark
void CustomArguments(benchmark::internal::Benchmark* b) {
    // Test different combinations of table sizes and partitions
    for (auto nranks : {4}) {
        for (int size_mb : {4000}) {
            for (auto partitions : {2, 8, 32, 128, 512, 1024}) {
                b->Args({nranks, size_mb, partitions});
            }
        }
    }
}

// Register the benchmark with custom arguments
BENCHMARK(BM_PartitionAndPackCurrentImpl)
    ->Apply(CustomArguments)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);


// Register the benchmark with custom arguments
BENCHMARK(BM_PartitionAndPack)
    ->Args({4000, 2})
    ->Args({4000, 8})
    ->Args({4000, 32})
    ->Args({4000, 128})
    ->Args({4000, 512})
    ->Args({4000, 1024})
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
