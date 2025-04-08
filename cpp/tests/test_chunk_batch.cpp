/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <rapidsmp/shuffler/chunk_batch.hpp>

namespace rapidsmp::shuffler::detail {

constexpr std::size_t operator""_KiB(unsigned long long n) {
    return n * (1 << 10);
}

// Allocate a buffer of the given size from the given resource.
std::unique_ptr<Buffer> allocate_buffer(
    MemoryType mem_type,
    std::size_t size,
    BufferResource& br,
    rmm::cuda_stream_view stream
) {
    auto [res, _] = br.reserve(mem_type, size, false);
    return br.allocate(mem_type, size, stream, res);
}

TEST(ChunkBatch, Empty) {
    uint32_t id = 1;
    Rank rank = 2;
    auto stream = rmm::cuda_stream_default;
    auto dev_mem_available = []() -> std::int64_t { return 1000_KiB; };
    BufferResource br{
        cudf::get_current_device_resource_ref(), {{MemoryType::DEVICE, dev_mem_available}}
    };

    auto test_empty_batch = [&](ChunkBatch& batch) {
        EXPECT_EQ(id, batch.id());
        EXPECT_EQ(rank, batch.destination());
        EXPECT_EQ(0, batch.size());

        auto chunks = batch.get_chunks(stream);
        EXPECT_EQ(0, chunks.size());
    };

    auto batch1 = ChunkBatch::create(id, rank, {}, &br, stream);
    test_empty_batch(batch1);

    // release the metadata buffer
    auto metadata = batch1.release_metadata();
    EXPECT_EQ(0, metadata->size());
}


}  // namespace rapidsmp::shuffler::detail
