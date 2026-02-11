/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>

#include <cudf/contiguous_split.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/pinned_memory_resource.hpp>

#include "utils/random_data.hpp"

#include <cucascade/memory/fixed_size_host_memory_resource.hpp>

using namespace cucascade::memory;

constexpr std::size_t MB = 1024 * 1024;
constexpr std::size_t GB = 1024 * MB;

void run_pack_pinned(
    benchmark::State& state,
    std::size_t table_size_bytes,
    rmm::device_async_resource_ref table_mr,
    rmm::device_async_resource_ref pack_mr,
    rmm::cuda_stream_view stream
) {
    auto const nrows =
        static_cast<cudf::size_type>(table_size_bytes / sizeof(random_data_t));
    auto table = random_table(1, nrows, 0, 1000, stream, table_mr);

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
    state.counters["table_size_mb"] =
        static_cast<double>(table_size_bytes) / static_cast<double>(MB);
    state.counters["num_rows"] = nrows;
    state.counters["bounce_buffer_mb"] = 0;
    state.counters["fixed_buffer_size_mb"] = 0;
    state.counters["num_blocks"] = 0;
    state.counters["batch_size"] = 0;
}

/**
 * @brief Benchmark packing a single 1GB table with rapidsmpf::PinnedMemoryResource
 */
static void BM_Pack_1GB_pinned_rapidsmpf(benchmark::State& state) {
    if (!rapidsmpf::is_pinned_memory_resources_supported()) {
        state.SkipWithMessage("Pinned memory resources are not supported");
        return;
    }

    constexpr std::size_t table_size_bytes = 1 * GB;
    rmm::cuda_stream_view stream = rmm::cuda_stream_default;

    rmm::mr::cuda_async_memory_resource cuda_mr;
    rapidsmpf::PinnedMemoryResource pinned_mr;

    run_pack_pinned(state, table_size_bytes, cuda_mr, pinned_mr, stream);
}

/**
 * @brief Benchmark packing a single 1GB table with rmm::mr::pinned_host_memory_resource
 */
static void BM_Pack_1GB_pinned_rmm(benchmark::State& state) {
    if (!rapidsmpf::is_pinned_memory_resources_supported()) {
        state.SkipWithMessage("Pinned memory resources are not supported");
        return;
    }

    constexpr std::size_t table_size_bytes = 1 * GB;
    rmm::cuda_stream_view stream = rmm::cuda_stream_default;

    rmm::mr::cuda_async_memory_resource cuda_mr;
    rmm::mr::pinned_host_memory_resource pinned_mr;

    run_pack_pinned(state, table_size_bytes, cuda_mr, pinned_mr, stream);
}

/**
 * @brief Benchmark chunked pack with fixed sized host buffers using cudaMemcpyAsync
 */
void run_chunked_pack_with_fixed_sized_pool_memcpy_async(
    benchmark::State& state,
    std::size_t bounce_buffer_size,
    std::size_t table_size,
    std::size_t fixed_buffer_size,
    rmm::device_async_resource_ref table_mr,
    rmm::device_async_resource_ref pack_mr,
    rmm::cuda_stream_view stream
) {
    // Calculate number of rows for a single-column table of the desired size
    auto const nrows = static_cast<cudf::size_type>(table_size / sizeof(random_data_t));
    auto table = random_table(1, nrows, 0, 1000, stream, table_mr);

    // Create the chunked_pack instance to get total output size
    size_t n_buffers;
    {
        cudf::chunked_pack packer(table.view(), bounce_buffer_size, stream, table_mr);
        // upper bound multiple of fixed buffer size
        n_buffers = (packer.get_total_contiguous_size() + fixed_buffer_size - 1)
                    / fixed_buffer_size;
    }

    // Setup fixed_size_host_memory_resource
    const int device_id = 0;
    const std::size_t mem_limit = 8 * GB;  // 8GB
    const std::size_t capacity = 8 * GB;  // 8GB
    const std::size_t pool_size = 128;  // 128 blocks
    const std::size_t initial_pools = 4;  // 4 pools

    rmm::mr::pinned_host_memory_resource upstream_mr;
    fixed_size_host_memory_resource host_mr(
        device_id,
        upstream_mr,
        mem_limit,
        capacity,
        fixed_buffer_size,
        pool_size,
        initial_pools
    );

    // Allocate fixed sized host buffers for the destination
    auto fixed_host_buffers =
        host_mr.allocate_multiple_blocks(n_buffers * fixed_buffer_size);

    // Allocate device bounce buffer
    rmm::device_buffer bounce_buffer(bounce_buffer_size, stream, pack_mr);
    cudf::device_span<std::uint8_t> bounce_buffer_span(
        static_cast<std::uint8_t*>(bounce_buffer.data()), bounce_buffer_size
    );

    auto run_packer = [&] {
        cudf::chunked_pack packer(table.view(), bounce_buffer_size, stream, pack_mr);

        auto blocks_span = fixed_host_buffers->get_blocks();
        auto block_it = blocks_span.begin();

        while (packer.has_next()) {
            auto const bytes_copied = packer.next(bounce_buffer_span);

            // Copy chunk to one or more fixed-size host blocks
            std::size_t offset = 0;
            while (offset < bytes_copied) {
                auto const copy_size = std::min(fixed_buffer_size, bytes_copied - offset);
                RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
                    *block_it,
                    static_cast<std::uint8_t const*>(bounce_buffer.data()) + offset,
                    copy_size,
                    cudaMemcpyDeviceToHost,
                    stream.value()
                ));

                offset += fixed_buffer_size;
                ++block_it;
            }
        }
    };

    {  // Warm up
        run_packer();
        stream.synchronize();
    }

    for (auto _ : state) {
        run_packer();
        benchmark::DoNotOptimize(fixed_host_buffers);
        benchmark::DoNotOptimize(bounce_buffer);
        stream.synchronize();
    }

    state.SetBytesProcessed(
        static_cast<std::int64_t>(state.iterations())
        * static_cast<std::int64_t>(table_size)
    );
    auto const batch_size = bounce_buffer_size / fixed_buffer_size;
    state.counters["table_size_mb"] =
        static_cast<double>(table_size) / static_cast<double>(MB);
    state.counters["num_rows"] = nrows;
    state.counters["bounce_buffer_mb"] =
        static_cast<double>(bounce_buffer_size) / static_cast<double>(MB);
    state.counters["fixed_buffer_size_mb"] =
        static_cast<double>(fixed_buffer_size) / static_cast<double>(MB);
    state.counters["num_blocks"] = n_buffers;
    state.counters["batch_size"] = batch_size;
}

/**
 * @brief Benchmark chunked pack with fixed sized host buffers using cudaMemcpyBatchAsync
 */
void run_chunked_pack_with_fixed_sized_pool_batch_async(
    benchmark::State& state,
    std::size_t bounce_buffer_size,
    std::size_t table_size,
    std::size_t fixed_buffer_size,
    rmm::device_async_resource_ref table_mr,
    rmm::device_async_resource_ref pack_mr,
    rmm::cuda_stream_view stream
) {
    // Calculate number of rows for a single-column table of the desired size
    auto const nrows = static_cast<cudf::size_type>(table_size / sizeof(random_data_t));
    auto table = random_table(1, nrows, 0, 1000, stream, table_mr);

    // Create the chunked_pack instance to get total output size
    size_t n_buffers;
    {
        cudf::chunked_pack packer(table.view(), bounce_buffer_size, stream, table_mr);
        // upper bound multiple of fixed buffer size
        n_buffers = (packer.get_total_contiguous_size() + fixed_buffer_size - 1)
                    / fixed_buffer_size;
    }

    // Setup fixed_size_host_memory_resource
    const int device_id = 0;
    const std::size_t mem_limit = 8ull * GB;  // 8GB
    const std::size_t capacity = 8ull * GB;  // 8GB
    const std::size_t pool_size = 128;  // 128 blocks
    const std::size_t initial_pools = 4;  // 4 pools

    rmm::mr::pinned_host_memory_resource upstream_mr;
    fixed_size_host_memory_resource host_mr(
        device_id,
        upstream_mr,
        mem_limit,
        capacity,
        fixed_buffer_size,
        pool_size,
        initial_pools
    );

    // Allocate fixed sized host buffers for the destination
    auto fixed_host_buffers =
        host_mr.allocate_multiple_blocks(n_buffers * fixed_buffer_size);

    // Allocate device bounce buffer
    rmm::device_buffer bounce_buffer(bounce_buffer_size, stream, pack_mr);
    cudf::device_span<std::uint8_t> bounce_buffer_span(
        static_cast<std::uint8_t*>(bounce_buffer.data()), bounce_buffer_size
    );

    // Max copies per iteration: one bounce buffer fills this many fixed buffers
    auto const n_copies_per_batch = bounce_buffer_size / fixed_buffer_size;

    auto run_packer = [&] {
        cudf::chunked_pack packer(table.view(), bounce_buffer_size, stream, pack_mr);

        // Get blocks span and use forward iterator
        auto blocks_span = fixed_host_buffers->get_blocks();
        auto block_it = blocks_span.begin();

        // Prepare batch copy arrays. Source pointers are fixed (bounce buffer layout);
        // destinations are taken from the blocks span (&*block_it). Only sizes vary per
        // chunk.
        auto const bounce_ptr = static_cast<std::uint8_t*>(bounce_buffer.data());
        std::vector<const void*> srcs(n_copies_per_batch);
        for (size_t i = 0; i < n_copies_per_batch; ++i) {
            srcs[i] = bounce_ptr + i * fixed_buffer_size;
        }
        std::vector<size_t> sizes(n_copies_per_batch);

        cudaMemcpyAttributes attrs{};
        attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
        std::array<size_t, 1> attrsIdxs{0};

        while (packer.has_next()) {
            auto const bytes_copied = packer.next(bounce_buffer_span);

            size_t const n_copies =
                (bytes_copied + fixed_buffer_size - 1) / fixed_buffer_size;

            if (n_copies > 0) {
                // Only sizes change per iteration: first n_copies-1 full, last is
                // remainder.
                std::fill_n(sizes.begin(), n_copies - 1, fixed_buffer_size);
                sizes[n_copies - 1] = bytes_copied - (n_copies - 1) * fixed_buffer_size;

                RAPIDSMPF_CUDA_TRY(cudaMemcpyBatchAsync(
                    reinterpret_cast<void* const*>(&*block_it),
                    srcs.data(),
                    sizes.data(),
                    n_copies,
                    &attrs,
                    attrsIdxs.data(),
                    attrsIdxs.size(),
                    stream.value()
                ));
                block_it += static_cast<std::ptrdiff_t>(n_copies);
            }
        }

        stream.synchronize();
    };

    {  // Warm up
        run_packer();
    }

    for (auto _ : state) {
        run_packer();
        benchmark::DoNotOptimize(fixed_host_buffers);
        benchmark::DoNotOptimize(bounce_buffer);
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
    state.counters["fixed_buffer_size_mb"] =
        static_cast<double>(fixed_buffer_size) / static_cast<double>(MB);
    state.counters["num_blocks"] = n_buffers;
    state.counters["batch_size"] = n_copies_per_batch;
}

/**
 * @brief Benchmark for chunked pack with fixed sized pool using cudaMemcpyAsync
 */
static void BM_ChunkedPack_FixedPool_MemcpyAsync(benchmark::State& state) {
    auto const bounce_buffer_mb = static_cast<std::size_t>(state.range(0));
    auto const bounce_buffer_size = bounce_buffer_mb * MB;
    auto const fixed_buffer_size = 1 * MB;
    constexpr std::size_t table_size_bytes = 1 * GB;  // 1GB table

    rmm::cuda_stream_view stream = rmm::cuda_stream_default;
    rmm::mr::cuda_async_memory_resource cuda_mr;

    run_chunked_pack_with_fixed_sized_pool_memcpy_async(
        state,
        bounce_buffer_size,
        table_size_bytes,
        fixed_buffer_size,
        cuda_mr,
        cuda_mr,
        stream
    );
}

/**
 * @brief Benchmark for chunked pack with fixed sized pool using cudaMemcpyBatchAsync
 */
static void BM_ChunkedPack_FixedPool_BatchAsync(benchmark::State& state) {
    auto const bounce_buffer_mb = static_cast<std::size_t>(state.range(0));
    auto const bounce_buffer_size = bounce_buffer_mb * MB;
    auto const fixed_buffer_size = 1 * MB;
    constexpr std::size_t table_size_bytes = 1 * GB;  // 1GB table

    // cudaMemcpyBatchAsync requires a non-legacy stream (not the default NULL stream)
    rmm::cuda_stream stream_obj;
    rmm::cuda_stream_view stream(stream_obj);
    rmm::mr::cuda_async_memory_resource cuda_mr;

    run_chunked_pack_with_fixed_sized_pool_batch_async(
        state,
        bounce_buffer_size,
        table_size_bytes,
        fixed_buffer_size,
        cuda_mr,
        cuda_mr,
        stream
    );
}

/**
 * @brief Custom argument generator for fixed pool benchmarks.
 */
void FixedPoolArguments(benchmark::internal::Benchmark* b) {
    // Test different bounce buffer sizes in MB: 1MB to 128MB
    for (auto bounce_buf_sz_mb : {1, 2, 4, 8, 16, 32, 64, 128}) {
        b->Args({bounce_buf_sz_mb});
    }
}

BENCHMARK(BM_Pack_1GB_pinned_rapidsmpf)->UseRealTime()->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Pack_1GB_pinned_rmm)->UseRealTime()->Unit(benchmark::kMillisecond);

BENCHMARK(BM_ChunkedPack_FixedPool_MemcpyAsync)
    ->Apply(FixedPoolArguments)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_ChunkedPack_FixedPool_BatchAsync)
    ->Apply(FixedPoolArguments)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
