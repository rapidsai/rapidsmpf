/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
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

#include <rapidsmpf/memory/host_buffer.hpp>
#include <rapidsmpf/memory/host_memory_resource.hpp>
#include <rapidsmpf/memory/pinned_memory_resource.hpp>

#include "utils/random_data.hpp"

constexpr std::size_t MB = 1024 * 1024;

/**
 * @brief Runs the cudf::pack using a device-accessible memory resource
 * @param state The benchmark state
 * @param table_size_mb The size of the table in MB
 * @param table_mr The memory resource for the table
 * @param pack_mr The memory resource for the packed data
 * @param stream The CUDA stream to use
 */
void run_pack(
    benchmark::State& state,
    std::size_t table_size_mb,
    rmm::device_async_resource_ref table_mr,
    rmm::device_async_resource_ref pack_mr,
    rmm::cuda_stream_view stream
) {
    auto const table_size_bytes = table_size_mb * MB;

    // Calculate number of rows for a single-column table of the desired size
    auto const nrows =
        static_cast<cudf::size_type>(table_size_bytes / sizeof(random_data_t));
    auto table = random_table(1, nrows, 0, 1000, stream, table_mr);

    // Warm up
    auto warm_up = cudf::pack(table.view(), stream, pack_mr);
    stream.synchronize();

    for (auto _ : state) {
        auto packed = cudf::pack(table.view(), stream, pack_mr);
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
 * @brief Benchmark for cudf::pack with device memory
 */
static void BM_Pack_device(benchmark::State& state) {
    auto const table_size_mb = static_cast<std::size_t>(state.range(0));

    rmm::cuda_stream_view stream = rmm::cuda_stream_default;

    // Create memory resources
    rmm::mr::cuda_async_memory_resource cuda_mr;
    rmm::mr::pool_memory_resource<rmm::mr::cuda_async_memory_resource> pool_mr{
        cuda_mr, rmm::percent_of_free_device_memory(40)
    };
    run_pack(state, table_size_mb, pool_mr, pool_mr, stream);
}

/**
 * @brief Benchmark for cudf::pack with device memory
 */
static void BM_Pack_pinned_host(benchmark::State& state) {
    if (!rapidsmpf::is_pinned_memory_resources_supported()) {
        state.SkipWithMessage("Pinned memory resources are not supported");
        return;
    }

    auto const table_size_mb = static_cast<std::size_t>(state.range(0));

    rmm::cuda_stream_view stream = rmm::cuda_stream_default;

    // Create memory resources
    rmm::mr::cuda_async_memory_resource cuda_mr;
    rmm::mr::pool_memory_resource<rmm::mr::cuda_async_memory_resource> pool_mr{
        cuda_mr, rmm::percent_of_free_device_memory(40)
    };
    rapidsmpf::PinnedMemoryResource pinned_mr;
    run_pack(state, table_size_mb, pool_mr, pinned_mr, stream);
}

/**
 * @brief Runs the cudf::pack and copy the packed data to a host buffer
 * @param state The benchmark state
 * @param table_size_mb The size of the table in MB
 * @param table_mr The memory resource for the table
 * @param pack_mr The memory resource for the packed data
 * @param stream The CUDA stream to use
 */
void run_pack_and_copy(
    benchmark::State& state,
    std::size_t table_size_mb,
    rmm::device_async_resource_ref table_mr,
    rmm::device_async_resource_ref pack_mr,
    rmm::host_async_resource_ref dest_mr,
    rmm::cuda_stream_view stream
) {
    auto const table_size_bytes = table_size_mb * MB;

    // Calculate number of rows for a single-column table of the desired size
    auto const nrows =
        static_cast<cudf::size_type>(table_size_bytes / sizeof(random_data_t));
    auto table = random_table(1, nrows, 0, 1000, stream, table_mr);

    // Warm up
    auto warm_up = cudf::pack(table.view(), stream, pack_mr);

    rapidsmpf::HostBuffer dest(warm_up.gpu_data->size(), stream, dest_mr);
    stream.synchronize();

    for (auto _ : state) {
        auto packed = cudf::pack(table.view(), stream, pack_mr);
        RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
            dest.data(),
            packed.gpu_data->data(),
            packed.gpu_data->size(),
            cudaMemcpyDefault,
            stream
        ));
        benchmark::DoNotOptimize(packed);
        benchmark::DoNotOptimize(dest);
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
 * @brief Benchmark for cudf::pack with device memory
 */
static void BM_Pack_device_copy_to_host(benchmark::State& state) {
    auto const table_size_mb = static_cast<std::size_t>(state.range(0));

    rmm::cuda_stream_view stream = rmm::cuda_stream_default;

    // Create memory resources
    rmm::mr::cuda_async_memory_resource cuda_mr;
    rmm::mr::pool_memory_resource<rmm::mr::cuda_async_memory_resource> pool_mr{
        cuda_mr, rmm::percent_of_free_device_memory(40)
    };
    rapidsmpf::HostMemoryResource host_mr;
    run_pack_and_copy(state, table_size_mb, pool_mr, pool_mr, host_mr, stream);
}

/**
 * @brief Benchmark for cudf::pack with device memory
 */
static void BM_Pack_device_copy_to_pinned_host(benchmark::State& state) {
    if (!rapidsmpf::is_pinned_memory_resources_supported()) {
        state.SkipWithMessage("Pinned memory resources are not supported");
        return;
    }

    auto const table_size_mb = static_cast<std::size_t>(state.range(0));

    rmm::cuda_stream_view stream = rmm::cuda_stream_default;

    // Create memory resources
    rmm::mr::cuda_async_memory_resource cuda_mr;
    rmm::mr::pool_memory_resource<rmm::mr::cuda_async_memory_resource> pool_mr{
        cuda_mr, rmm::percent_of_free_device_memory(40)
    };
    rapidsmpf::PinnedMemoryResource pinned_mr;
    run_pack_and_copy(state, table_size_mb, pool_mr, pool_mr, pinned_mr, stream);
}

/**
 * @brief Benchmark for cudf::pack with pinned memory
 */
static void BM_Pack_pinned_copy_to_host(benchmark::State& state) {
    if (!rapidsmpf::is_pinned_memory_resources_supported()) {
        state.SkipWithMessage("Pinned memory resources are not supported");
        return;
    }

    auto const table_size_mb = static_cast<std::size_t>(state.range(0));

    rmm::cuda_stream_view stream = rmm::cuda_stream_default;

    // Create memory resources
    rmm::mr::cuda_async_memory_resource cuda_mr;
    rmm::mr::pool_memory_resource<rmm::mr::cuda_async_memory_resource> pool_mr{
        cuda_mr, rmm::percent_of_free_device_memory(40)
    };
    rapidsmpf::PinnedMemoryResource pinned_mr;
    rapidsmpf::HostMemoryResource dest_mr;

    run_pack_and_copy(state, table_size_mb, pool_mr, pinned_mr, dest_mr, stream);
}

/**
 * @brief Runs the cudf::chunked_pack benchmark
 * @param state The benchmark state
 * @param bounce_buffer_size The size of the bounce buffer in bytes
 * @param table_size The size of the table in bytes
 * @param table_mr The memory resource for the table
 * @param pack_mr The memory resource for the packed data
 * @param dest_mr The memory resource for the destination data
 * @param stream The CUDA stream to use
 *
 * @tparam DestinationBufferType The type of the destination buffer
 */
template <typename DestinationBufferType = rapidsmpf::HostBuffer>
void run_chunked_pack(
    benchmark::State& state,
    std::size_t bounce_buffer_size,
    std::size_t table_size,
    rmm::device_async_resource_ref table_mr,
    rmm::device_async_resource_ref pack_mr,
    auto& dest_mr,
    rmm::cuda_stream_view stream
) {
    // Calculate number of rows for a single-column table of the desired size
    auto const nrows = static_cast<cudf::size_type>(table_size / sizeof(random_data_t));
    auto table = random_table(1, nrows, 0, 1000, stream, table_mr);

    // Create the chunked_pack instance to get total output size
    size_t total_size;
    {
        cudf::chunked_pack packer(table.view(), bounce_buffer_size, stream, table_mr);
        total_size = packer.get_total_contiguous_size();
    }

    // Allocate bounce buffer and destination buffer using the pack_mr
    rmm::device_buffer bounce_buffer(bounce_buffer_size, stream, pack_mr);
    DestinationBufferType destination(total_size, stream, dest_mr);

    auto run_packer = [&] {
        cudf::chunked_pack packer(table.view(), bounce_buffer_size, stream, pack_mr);

        std::size_t offset = 0;
        while (packer.has_next()) {
            auto const bytes_copied = packer.next(
                cudf::device_span<std::uint8_t>(
                    static_cast<std::uint8_t*>(bounce_buffer.data()), bounce_buffer_size
                )
            );
            RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
                static_cast<std::byte*>(destination.data()) + offset,
                bounce_buffer.data(),
                bytes_copied,
                cudaMemcpyDefault,
                stream.value()
            ));
            offset += bytes_copied;
        }
    };

    {  // Warm up
        run_packer();
        stream.synchronize();
    }

    for (auto _ : state) {
        run_packer();
        benchmark::DoNotOptimize(destination);
        stream.synchronize();
    }

    state.SetBytesProcessed(
        static_cast<std::int64_t>(state.iterations())
        * static_cast<std::int64_t>(table_size)
    );
    state.counters["table_size_mb"] =
        static_cast<double>(table_size) / static_cast<double>(MB);
    state.counters["num_rows"] = nrows;
    state.counters["bounce_buffer_mb"] =
        static_cast<double>(bounce_buffer_size) / static_cast<double>(MB);
}

/**
 * @brief Benchmark for cudf::chunked_pack with device bounce buffer and destination
 * buffer.
 * @param state The benchmark state containing the table size in MB as the first range
 * argument.
 */
static void BM_ChunkedPack_device_copy_to_device(benchmark::State& state) {
    auto const table_size_mb = static_cast<std::size_t>(state.range(0));
    auto const table_size_bytes = table_size_mb * MB;

    // Bounce buffer size: max(1MB, table_size / 10)
    auto const bounce_buffer_size = std::max(MB, table_size_bytes / 10);

    rmm::cuda_stream_view stream = rmm::cuda_stream_default;

    rmm::mr::cuda_async_memory_resource cuda_mr;
    rmm::mr::pool_memory_resource<rmm::mr::cuda_async_memory_resource> pool_mr{
        cuda_mr, rmm::percent_of_free_device_memory(40)
    };

    run_chunked_pack<rmm::device_buffer>(
        state, bounce_buffer_size, table_size_bytes, pool_mr, pool_mr, pool_mr, stream
    );
}

/**
 * @brief Benchmark for cudf::chunked_pack with device bounce buffer and pinned host
 * destination buffer.
 */
static void BM_ChunkedPack_device_copy_to_pinned_host(benchmark::State& state) {
    if (!rapidsmpf::is_pinned_memory_resources_supported()) {
        state.SkipWithMessage("Pinned memory resources are not supported");
        return;
    }

    auto const table_size_mb = static_cast<std::size_t>(state.range(0));
    auto const table_size_bytes = table_size_mb * MB;

    // Bounce buffer size: max(1MB, table_size / 10)
    auto const bounce_buffer_size = std::max(MB, table_size_bytes / 10);

    rmm::cuda_stream_view stream = rmm::cuda_stream_default;

    rmm::mr::cuda_async_memory_resource cuda_mr;
    rmm::mr::pool_memory_resource<rmm::mr::cuda_async_memory_resource> pool_mr{
        cuda_mr, rmm::percent_of_free_device_memory(40)
    };
    rapidsmpf::PinnedMemoryResource pinned_mr;

    run_chunked_pack(
        state, bounce_buffer_size, table_size_bytes, pool_mr, pool_mr, pinned_mr, stream
    );
}

/**
 * @brief Benchmark for cudf::chunked_pack with device bounce buffer and host destination
 * buffer.
 * @param state The benchmark state containing the table size in MB as the first range
 * argument.
 */
static void BM_ChunkedPack_device_copy_to_host(benchmark::State& state) {
    auto const table_size_mb = static_cast<std::size_t>(state.range(0));
    auto const table_size_bytes = table_size_mb * MB;

    // Bounce buffer size: max(1MB, table_size / 10)
    auto const bounce_buffer_size = std::max(MB, table_size_bytes / 10);

    rmm::cuda_stream_view stream = rmm::cuda_stream_default;

    rmm::mr::cuda_async_memory_resource cuda_mr;
    rmm::mr::pool_memory_resource<rmm::mr::cuda_async_memory_resource> pool_mr{
        cuda_mr, rmm::percent_of_free_device_memory(40)
    };
    rapidsmpf::HostMemoryResource host_mr;

    run_chunked_pack(
        state, bounce_buffer_size, table_size_bytes, pool_mr, pool_mr, host_mr, stream
    );
}

/**
 * @brief Benchmark for cudf::chunked_pack with pinned bounce buffer and pinned host
 * destination buffer. memory.
 * @param state The benchmark state containing the table size in MB as the first range
 * argument.
 */
static void BM_ChunkedPack_pinned_copy_to_pinned_host(benchmark::State& state) {
    if (!rapidsmpf::is_pinned_memory_resources_supported()) {
        state.SkipWithMessage("Pinned memory resources are not supported");
        return;
    }

    auto const table_size_mb = static_cast<std::size_t>(state.range(0));
    auto const table_size_bytes = table_size_mb * MB;

    // Bounce buffer size: max(1MB, table_size / 10)
    auto const bounce_buffer_size = std::max(MB, table_size_bytes / 10);

    rmm::cuda_stream_view stream = rmm::cuda_stream_default;

    rmm::mr::cuda_async_memory_resource cuda_mr;
    rmm::mr::pool_memory_resource<rmm::mr::cuda_async_memory_resource> pool_mr{
        cuda_mr, rmm::percent_of_free_device_memory(40)
    };
    rapidsmpf::PinnedMemoryResource pinned_mr;

    run_chunked_pack(
        state, bounce_buffer_size, table_size_bytes, pool_mr, pinned_mr, pinned_mr, stream
    );
}

/**
 * @brief Benchmark for cudf::chunked_pack with pinned bounce buffer and host destination
 * buffer.
 * @param state The benchmark state containing the table size in MB as the first range
 * argument.
 */
static void BM_ChunkedPack_pinned_copy_to_host(benchmark::State& state) {
    if (!rapidsmpf::is_pinned_memory_resources_supported()) {
        state.SkipWithMessage("Pinned memory resources are not supported");
        return;
    }

    auto const table_size_mb = static_cast<std::size_t>(state.range(0));
    auto const table_size_bytes = table_size_mb * MB;

    // Bounce buffer size: max(1MB, table_size / 10)
    auto const bounce_buffer_size = std::max(MB, table_size_bytes / 10);

    rmm::cuda_stream_view stream = rmm::cuda_stream_default;

    rmm::mr::cuda_async_memory_resource cuda_mr;
    rmm::mr::pool_memory_resource<rmm::mr::cuda_async_memory_resource> pool_mr{
        cuda_mr, rmm::percent_of_free_device_memory(40)
    };
    rapidsmpf::PinnedMemoryResource pinned_mr;
    rapidsmpf::HostMemoryResource host_mr;

    run_chunked_pack(
        state, bounce_buffer_size, table_size_bytes, pool_mr, pinned_mr, host_mr, stream
    );
}

/**
 * @brief Custom argument generator for pack benchmarks.
 *
 * Configures benchmarks to run with various table sizes ranging from 1MB to 4GB.
 *
 * @param b The benchmark to configure with arguments.
 */
void PackArguments(benchmark::internal::Benchmark* b) {
    // Test different table sizes in MB (minimum 1MB as requested)
    for (auto size_mb : {1, 10, 100, 500, 1000, 2000, 4000}) {
        b->Args({size_mb});
    }
}

// Register the benchmarks

// Pack benchmarks
BENCHMARK(BM_Pack_device)
    ->Apply(PackArguments)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Pack_pinned_host)
    ->Apply(PackArguments)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

// Pack and copy benchmarks
BENCHMARK(BM_Pack_device_copy_to_host)
    ->Apply(PackArguments)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Pack_device_copy_to_pinned_host)
    ->Apply(PackArguments)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Pack_pinned_copy_to_host)
    ->Apply(PackArguments)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

// Chunked pack benchmarks

BENCHMARK(BM_ChunkedPack_device_copy_to_device)
    ->Apply(PackArguments)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_ChunkedPack_device_copy_to_pinned_host)
    ->Apply(PackArguments)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_ChunkedPack_device_copy_to_host)
    ->Apply(PackArguments)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_ChunkedPack_pinned_copy_to_pinned_host)
    ->Apply(PackArguments)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_ChunkedPack_pinned_copy_to_host)
    ->Apply(PackArguments)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

/**
 * @brief Benchmark for cudf::chunked_pack in device memory varying the bounce buffer size
 * and keeping table size fixed at 1GB
 */
static void BM_ChunkedPack_fixed_table_device(benchmark::State& state) {
    auto const bounce_buffer_size = static_cast<std::size_t>(state.range(0)) * MB;
    constexpr std::size_t table_size_bytes = 1024 * MB;

    rmm::cuda_stream_view stream = rmm::cuda_stream_default;

    // Create memory resources
    rmm::mr::cuda_async_memory_resource cuda_mr;

    rmm::mr::pool_memory_resource<rmm::mr::cuda_async_memory_resource> pool_mr{
        cuda_mr, rmm::percent_of_free_device_memory(40)
    };

    run_chunked_pack<rmm::device_buffer>(
        state, bounce_buffer_size, table_size_bytes, pool_mr, pool_mr, pool_mr, stream
    );
}

/**
 * @brief Benchmark for cudf::chunked_pack in pinned memory varying the bounce buffer size
 * and keeping table size fixed at 1GB
 */
static void BM_ChunkedPack_fixed_table_pinned(benchmark::State& state) {
    if (!rapidsmpf::is_pinned_memory_resources_supported()) {
        state.SkipWithMessage("Pinned memory resources are not supported");
        return;
    }

    auto const bounce_buffer_size = static_cast<std::size_t>(state.range(0)) * MB;
    constexpr std::size_t table_size_bytes = 1024 * MB;

    rmm::cuda_stream_view stream = rmm::cuda_stream_default;

    rmm::mr::cuda_async_memory_resource cuda_mr;
    rmm::mr::pool_memory_resource<rmm::mr::cuda_async_memory_resource> pool_mr{
        cuda_mr, rmm::percent_of_free_device_memory(40)
    };
    rapidsmpf::PinnedMemoryResource pinned_mr;

    run_chunked_pack<rmm::device_buffer>(
        state, bounce_buffer_size, table_size_bytes, pool_mr, pinned_mr, pinned_mr, stream
    );
}

/**
 * @brief Custom argument generator for chunked pack benchmarks with fixed table size.
 *
 * Configures benchmarks to run with various bounce buffer sizes ranging from 1MB to 1GB.
 *
 * @param b The benchmark to configure with arguments.
 */
void ChunkedPackArguments(benchmark::internal::Benchmark* b) {
    // Test different bounce buffer sizes in MB
    for (auto bounce_buf_sz_mb : {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}) {
        b->Args({bounce_buf_sz_mb});
    }
}

BENCHMARK(BM_ChunkedPack_fixed_table_device)
    ->Apply(ChunkedPackArguments)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_ChunkedPack_fixed_table_pinned)
    ->Apply(ChunkedPackArguments)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
