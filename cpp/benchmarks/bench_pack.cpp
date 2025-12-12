/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstdint>
#include <memory>

#include <benchmark/benchmark.h>

#include <cudf/contiguous_split.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include "utils/random_data.hpp"

constexpr std::size_t MB = 1024 * 1024;

/**
 * @brief Benchmark for cudf::pack
 *
 * Measures the time to pack a table into contiguous memory using cudf::pack.
 */
static void BM_Pack(benchmark::State& state) {
    auto const table_size_mb = static_cast<std::size_t>(state.range(0));
    auto const table_size_bytes = table_size_mb * MB;

    rmm::cuda_stream_view stream = rmm::cuda_stream_default;

    // Create memory resources
    rmm::mr::cuda_async_memory_resource cuda_mr;

    rmm::mr::pool_memory_resource<rmm::mr::cuda_async_memory_resource> pool_mr{
        cuda_mr, rmm::percent_of_free_device_memory(40)
    };

    // Calculate number of rows for a single-column table of the desired size
    auto const nrows =
        static_cast<cudf::size_type>(table_size_bytes / sizeof(random_data_t));
    auto table = random_table(1, nrows, 0, 1000, stream, pool_mr);

    // Warm up
    auto warm_up = cudf::pack(table.view(), stream, pool_mr);
    stream.synchronize();

    for (auto _ : state) {
        auto packed = cudf::pack(table.view(), stream, pool_mr);
        benchmark::DoNotOptimize(packed);
        stream.synchronize();
    }

    state.SetBytesProcessed(
        static_cast<std::int64_t>(state.iterations())
        * static_cast<std::int64_t>(table_size_bytes)
    );
    state.counters["table_size_mb"] = static_cast<double>(table_size_mb);
    state.counters["num_rows"] = nrows;
}

/**
 * @brief Benchmark for cudf::chunked_pack
 *
 * Measures the time to pack a table into contiguous memory using cudf::chunked_pack.
 * The bounce buffer size is set to max(1MB, table_size / 10).
 */
static void BM_ChunkedPack(benchmark::State& state) {
    auto const table_size_mb = static_cast<std::size_t>(state.range(0));
    auto const table_size_bytes = table_size_mb * MB;

    // Bounce buffer size: max(1MB, table_size / 10)
    auto const bounce_buffer_size = std::max(MB, table_size_bytes / 10);

    rmm::cuda_stream_view stream = rmm::cuda_stream_default;

    // Create memory resources
    rmm::mr::cuda_async_memory_resource cuda_mr;

    rmm::mr::pool_memory_resource<rmm::mr::cuda_async_memory_resource> pool_mr{
        cuda_mr, rmm::percent_of_free_device_memory(40)
    };

    // Calculate number of rows for a single-column table of the desired size
    auto const nrows =
        static_cast<cudf::size_type>(table_size_bytes / sizeof(random_data_t));
    auto table = random_table(1, nrows, 0, 1000, stream, pool_mr);

    // Create the chunked_pack instance to get total output size
    cudf::chunked_pack packer(table.view(), bounce_buffer_size, stream, pool_mr);
    auto const total_size = packer.get_total_contiguous_size();

    // Allocate bounce buffer and destination buffer
    rmm::device_buffer bounce_buffer(bounce_buffer_size, stream, pool_mr);
    rmm::device_buffer destination(total_size, stream, pool_mr);

    auto run_packer = [&](cudf::chunked_pack& packer) {
        std::size_t offset = 0;
        while (packer.has_next()) {
            auto const bytes_copied = packer.next(cudf::device_span<std::uint8_t>(
                static_cast<std::uint8_t*>(bounce_buffer.data()), bounce_buffer_size
            ));
            RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
                static_cast<std::uint8_t*>(destination.data()) + offset,
                bounce_buffer.data(),
                bytes_copied,
                cudaMemcpyDefault,
                stream.value()
            ));
            offset += bytes_copied;
        }
    };

    // Warm up
    {
        cudf::chunked_pack packer(table.view(), bounce_buffer_size, stream, pool_mr);
        run_packer(packer);
        stream.synchronize();
    }

    for (auto _ : state) {
        cudf::chunked_pack packer(table.view(), bounce_buffer_size, stream, pool_mr);
        run_packer(packer);
        benchmark::DoNotOptimize(destination);
        stream.synchronize();
    }

    state.SetBytesProcessed(
        static_cast<std::int64_t>(state.iterations())
        * static_cast<std::int64_t>(table_size_bytes)
    );
    state.counters["table_size_mb"] = static_cast<double>(table_size_mb);
    state.counters["num_rows"] = nrows;
    state.counters["bounce_buffer_mb"] =
        static_cast<double>(bounce_buffer_size) / static_cast<double>(MB);
}

// /**
//  * @brief Benchmark for cudf::chunked_pack without the extra device-to-device copy
//  *
//  * Measures the time to pack a table into contiguous memory using cudf::chunked_pack,
//  * writing directly to the destination buffer without an intermediate bounce buffer
//  copy.
//  */
// static void BM_ChunkedPackDirect(benchmark::State& state) {
//     auto const table_size_mb = static_cast<std::size_t>(state.range(0));
//     auto const table_size_bytes = table_size_mb * MB;

//     // Bounce buffer size: max(1MB, table_size / 10)
//     auto const bounce_buffer_size = std::max(MB, table_size_bytes / 10);

//     rmm::cuda_stream_view stream = rmm::cuda_stream_default;

//     // Create memory resources
//     auto cuda_mr = std::make_unique<rmm::mr::cuda_memory_resource>();

//     cudaDeviceProp prop;
//     cudaGetDeviceProperties(&prop, 0);
//     auto const pool_size = static_cast<std::size_t>(prop.totalGlobalMem * 0.5);

//     auto pool_mr =
//         std::make_unique<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>(
//             cuda_mr.get(), pool_size
//         );

//     // Calculate number of rows for a single-column table of the desired size
//     auto const nrows =
//         static_cast<cudf::size_type>(table_size_bytes / sizeof(random_data_t));
//     auto table = random_table(1, nrows, 0, 1000, stream, pool_mr.get());

//     // Create the chunked_pack instance to get total output size
//     auto chunked_packer =
//         cudf::chunked_pack::create(table.view(), bounce_buffer_size, pool_mr.get());
//     auto const total_size = chunked_packer->build_metadata();

//     // Allocate destination buffer
//     rmm::device_buffer destination(total_size, stream, pool_mr.get());

//     // Warm up
//     {
//         auto packer =
//             cudf::chunked_pack::create(table.view(), bounce_buffer_size,
//             pool_mr.get());
//         packer->build_metadata();
//         std::size_t offset = 0;
//         while (packer->has_next()) {
//             auto const bytes_copied = packer->next(cudf::device_span<std::uint8_t>(
//                 static_cast<std::uint8_t*>(destination.data()) + offset,
//                 std::min(bounce_buffer_size, total_size - offset)
//             ));
//             offset += bytes_copied;
//         }
//         stream.synchronize();
//     }

//     for (auto _ : state) {
//         auto packer =
//             cudf::chunked_pack::create(table.view(), bounce_buffer_size,
//             pool_mr.get());
//         packer->build_metadata();

//         std::size_t offset = 0;
//         while (packer->has_next()) {
//             auto const bytes_copied = packer->next(cudf::device_span<std::uint8_t>(
//                 static_cast<std::uint8_t*>(destination.data()) + offset,
//                 std::min(bounce_buffer_size, total_size - offset)
//             ));
//             offset += bytes_copied;
//         }
//         benchmark::DoNotOptimize(destination);
//         stream.synchronize();
//     }

//     state.SetBytesProcessed(
//         static_cast<std::int64_t>(state.iterations())
//         * static_cast<std::int64_t>(table_size_bytes)
//     );
//     state.counters["table_size_mb"] = static_cast<double>(table_size_mb);
//     state.counters["num_rows"] = nrows;
//     state.counters["bounce_buffer_mb"] =
//         static_cast<double>(bounce_buffer_size) / static_cast<double>(MB);
// }

// Custom argument generator for the benchmark
void PackArguments(benchmark::internal::Benchmark* b) {
    // Test different table sizes in MB (minimum 1MB as requested)
    for (auto size_mb : {1, 10, 100, 500, 1000, 2000, 4000}) {
        b->Args({size_mb});
    }
}

// Register the benchmarks
BENCHMARK(BM_Pack)->Apply(PackArguments)->UseRealTime()->Unit(benchmark::kMillisecond);

BENCHMARK(BM_ChunkedPack)
    ->Apply(PackArguments)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

// BENCHMARK(BM_ChunkedPackDirect)
//     ->Apply(PackArguments)
//     ->UseRealTime()
//     ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
