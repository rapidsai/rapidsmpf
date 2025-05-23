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

#include <rapidsmpf/shuffler/partition.hpp>

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
    const cudf::size_type num_rows = state.range(0);
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
        benchmark::DoNotOptimize(rapidsmpf::shuffler::partition_and_pack(
            *table,
            columns_to_hash,
            num_partitions,
            cudf::hash_id::HASH_MURMUR3,
            cudf::DEFAULT_HASH_SEED,
            stream,
            *pool_mr
        ));
        cudaStreamSynchronize(stream);
    }

    // Set metrics
    state.SetItemsProcessed(state.iterations() * int64_t(num_rows));
    state.SetBytesProcessed(
        state.iterations() * int64_t(num_rows) * int64_t(sizeof(int32_t))
    );
}


// Custom argument generator for the benchmark
void CustomArguments(benchmark::internal::Benchmark* b) {
    // Test different combinations of table sizes and partitions
    for (auto size : {1000000, 10000000, 100000000, 1000000000}) {  // 1M, 10M, 100M rows
        for (auto partitions : {2, 8, 100, 1000}) {
            b->Args({size, partitions});
        }
    }
}

// Register the benchmark with custom arguments
BENCHMARK(BM_PartitionAndPack)
    ->Apply(CustomArguments)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();