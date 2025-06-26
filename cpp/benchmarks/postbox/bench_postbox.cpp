/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <atomic>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

#include <benchmark/benchmark.h>

#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/host/pinned_memory_resource.hpp>

#include <rapidsmpf/buffer/buffer.hpp>
#include <rapidsmpf/buffer/packed_data.hpp>
#include <rapidsmpf/buffer/resource.hpp>
#include <rapidsmpf/shuffler/chunk.hpp>
#include <rapidsmpf/shuffler/postbox.hpp>

#include "postbox2.hpp"

namespace rapidsmpf::shuffler::detail {

// Helper function to create a chunk with device data buffer
Chunk create_chunk_with_device_data(
    ChunkID chunk_id,
    PartID part_id,
    size_t data_size,
    rmm::cuda_stream_view stream,
    BufferResource* br
) {
    // Create metadata
    auto metadata =
        std::make_unique<std::vector<uint8_t>>(std::vector<uint8_t>{1, 2, 3, 4});

    // Create device data buffer
    auto device_buffer = std::make_unique<rmm::device_buffer>(data_size, stream);
    std::vector<uint8_t> host_data(data_size);
    for (size_t i = 0; i < data_size; ++i) {
        host_data[i] = static_cast<uint8_t>(i % 256);
    }
    RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
        device_buffer->data(), host_data.data(), data_size, cudaMemcpyHostToDevice, stream
    ));
    auto event = std::make_shared<Buffer::Event>(stream);

    PackedData packed_data{std::move(metadata), std::move(device_buffer)};

    return Chunk::from_packed_data(
        chunk_id, part_id, std::move(packed_data), std::move(event), stream, br
    );
}

constexpr size_t nranks = 8;
constexpr size_t nparts = 100;

template <typename T>
T KeyMapFn(PartID pid) {
    return {};
}

template <>
Rank KeyMapFn<Rank>(PartID pid) {
    return pid % nranks;
};

template <>
PartID KeyMapFn<PartID>(PartID pid) {
    return pid;
};

// Benchmark template for PostBox
template <typename PostBoxType>
static void BM_PostBoxMultiThreaded(benchmark::State& state) {
    const auto num_chunks = static_cast<size_t>(state.range(0));
    const auto data_size = static_cast<size_t>(state.range(1));

    // Setup
    auto device_mr = std::make_unique<rmm::mr::cuda_memory_resource>();
    auto br = std::make_unique<BufferResource>(
        *device_mr,
        std::unordered_map<MemoryType, BufferResource::MemoryAvailable>{},
        std::nullopt
    );
    auto stream = cudf::get_default_stream();

    // Create PostBox with identity key mapping
    PostBoxType postbox(KeyMapFn<typename PostBoxType::key_type>, num_chunks);

    // Synchronization primitives
    std::atomic<size_t> chunks_inserted{0};
    std::atomic<size_t> chunks_extracted{0};

    auto insertion_finished = [&] { return chunks_inserted.load() == num_chunks; };
    auto extract_finished = [&] { return chunks_extracted.load() == num_chunks; };

    int64_t search_and_insert_count = 0;

    for (auto _ : state) {
        // Reset counters
        chunks_inserted = 0;
        chunks_extracted = 0;

        // Thread 1: Insert chunks with device data buffers
        std::thread insert_thread([&]() {
            for (size_t i = 0; i < num_chunks; ++i) {
                auto chunk = create_chunk_with_device_data(
                    i, static_cast<PartID>(i % nparts), data_size, stream, br.get()
                );
                postbox.insert(std::move(chunk));
                chunks_inserted.fetch_add(1);
            }
        });

        // Thread 2: Search for device data buffers, extract half, modify and reinsert
        std::thread search_thread([&]() {
            while (!insertion_finished() || !extract_finished()) {
                // Search for device data buffers
                auto device_chunks = postbox.search(MemoryType::DEVICE);

                // Extract half of them
                size_t extract_count = device_chunks.size() / 2;
                for (size_t i = 0; i < extract_count && i < device_chunks.size(); ++i) {
                    const auto& [key, cid, size] = device_chunks[i];

                    try {
                        auto chunk = postbox.extract(static_cast<PartID>(key), cid);

                        // Release device data buffer and set empty host data buffer
                        auto released = chunk.release_data_buffer();
                        benchmark::DoNotOptimize(released);
                        auto [reservation, overbooking] =
                            br->reserve(MemoryType::HOST, data_size, false);
                        auto host_buffer = br->allocate(
                            MemoryType::HOST, data_size, stream, reservation
                        );
                        std::memset(host_buffer->data(), 0, data_size);
                        chunk.set_data_buffer(std::move(host_buffer));

                        // Reinsert the chunk
                        postbox.insert(std::move(chunk));

                        search_and_insert_count++;
                    } catch (const std::out_of_range&) {
                        // Chunk was already extracted by another thread
                        continue;
                    }
                }
            }
        });

        // Thread 3: Extract all ready chunks
        std::thread extract_thread([&]() {
            while (!extract_finished()) {
                auto ready_chunks = postbox.extract_all_ready();
                chunks_extracted.fetch_add(ready_chunks.size());
                benchmark::DoNotOptimize(ready_chunks);
            }
        });

        // Wait for all threads to finish
        insert_thread.join();
        search_thread.join();
        extract_thread.join();

        // Verify all chunks were processed
        if (chunks_inserted.load() != num_chunks || chunks_extracted.load() != num_chunks)
        {
            state.SkipWithError("Not all chunks were processed");
        }
    }
    // state.counters["avg_search_and_insert_count"] = search_and_insert_count /
    // state.iterations();

    state.SetBytesProcessed(
        int64_t(state.iterations()) * int64_t(num_chunks) * int64_t(data_size)
    );
    state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(num_chunks));
}

// Warm up run 
BENCHMARK_TEMPLATE(BM_PostBoxMultiThreaded, PostBox<PartID>)
    ->Args({10000, 1024})  // 10k chunks, 1KB each
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

// Register benchmarks for PostBox
BENCHMARK_TEMPLATE(BM_PostBoxMultiThreaded, PostBox<PartID>)
    ->Args({10000, 1024})  // 10k chunks, 1KB each
    ->Args({100000, 1024})  // 100k chunks, 1KB each
    ->Args({1000000, 1024})  // 1M chunks, 1KB each
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(BM_PostBoxMultiThreaded, PostBox<Rank>)
    ->Args({10000, 1024})  // 10k chunks, 1KB each
    ->Args({100000, 1024})  // 100k chunks, 1KB each
    ->Args({1000000, 1024})  // 1M chunks, 1KB each
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);


// Register benchmarks for PostBox2
BENCHMARK_TEMPLATE(BM_PostBoxMultiThreaded, PostBox2<PartID>)
    ->Args({10000, 1024})  // 10k chunks, 1KB each
    ->Args({100000, 1024})  // 100k chunks, 1KB each
    ->Args({1000000, 1024})  // 1M chunks, 1KB each
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(BM_PostBoxMultiThreaded, PostBox2<Rank>)
    ->Args({10000, 1024})  // 10k chunks, 1KB each
    ->Args({100000, 1024})  // 100k chunks, 1KB each
    ->Args({1000000, 1024})  // 1M chunks, 1KB each
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);


}  // namespace rapidsmpf::shuffler::detail

BENCHMARK_MAIN();