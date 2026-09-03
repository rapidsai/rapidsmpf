/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <filesystem>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <utility>

#include <driver_types.h>
#include <gtest/gtest.h>

#include <cuda/stream>

#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/cuda_memcpy_async.hpp>
#include <rapidsmpf/memory/host_buffer.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/shuffler/chunk.hpp>
#include <rapidsmpf/shuffler/postbox.hpp>

#include "utils.hpp"

using namespace rapidsmpf;
using namespace rapidsmpf::shuffler;
using namespace rapidsmpf::shuffler::detail;

class ChunkTest : public ::testing::Test {
  protected:
    void SetUp() override {
        br = BufferResource::create(rmm::mr::get_current_device_resource_ref());
        stream = cuda::stream_ref{cudaStreamLegacy};
    }

    std::shared_ptr<BufferResource> br;
    cuda::stream_ref stream{cudaStreamLegacy};
};

TEST_F(ChunkTest, FromFinishedPartition) {
    ChunkID chunk_id = 123;
    PartID part_id = 456;
    std::size_t expected_num_chunks = 789;

    auto test_chunk = [&](Chunk& chunk) {
        EXPECT_EQ(chunk.chunk_id(), chunk_id);
        EXPECT_EQ(chunk.part_id(), part_id);
        EXPECT_EQ(chunk.expected_num_chunks(), expected_num_chunks);
        EXPECT_TRUE(chunk.is_control_message());
        EXPECT_EQ(chunk.metadata_size(), 0);
        EXPECT_EQ(chunk.data_size(), 0);
    };

    auto chunk = Chunk::from_finished_partition(chunk_id, part_id, expected_num_chunks);
    test_chunk(chunk);

    auto msg = chunk.serialize();
    auto chunk2 = Chunk::deserialize(*msg, br.get(), true);
    test_chunk(chunk2);
}

class ChunkFromPackedDataTest : public ChunkTest,
                                public ::testing::WithParamInterface<std::size_t> {};

TEST_P(ChunkFromPackedDataTest, RoundTrip) {
    std::size_t const data_size = GetParam();
    ChunkID chunk_id = 123;
    PartID part_id = 456;

    auto metadata = std::make_unique<std::vector<std::uint8_t>>(
        std::vector<std::uint8_t>{1, 2, 3, 4}
    );

    auto data = std::make_unique<rmm::device_buffer>(data_size, stream);
    if (data_size > 0) {
        std::vector<std::uint8_t> host_data(data_size);
        std::iota(host_data.begin(), host_data.end(), std::uint8_t{5});
        RAPIDSMPF_CUDA_TRY(
            cudaMemcpy(data->data(), host_data.data(), data_size, cudaMemcpyDefault)
        );
    }

    PackedData packed_data{std::move(metadata), br->move(std::move(data), stream)};

    auto test_chunk = [&](Chunk& chunk) {
        EXPECT_EQ(chunk.chunk_id(), chunk_id);
        EXPECT_EQ(chunk.part_id(), part_id);
        EXPECT_EQ(chunk.expected_num_chunks(), 0);
        EXPECT_FALSE(chunk.is_control_message());
        EXPECT_EQ(chunk.metadata_size(), 4);
        EXPECT_EQ(chunk.data_size(), data_size);
        EXPECT_TRUE(chunk.is_data_buffer_set());
    };

    auto chunk = Chunk::from_packed_data(chunk_id, part_id, std::move(packed_data));
    test_chunk(chunk);

    auto msg = chunk.serialize();
    auto chunk2 = Chunk::deserialize(*msg, br.get(), true);
    test_chunk(chunk2);
}

INSTANTIATE_TEST_SUITE_P(
    ChunkFromPackedData, ChunkFromPackedDataTest, ::testing::Values(0, 4)
);

namespace {

Chunk make_device_chunk(
    BufferResource& br, cuda::stream_ref stream, std::size_t data_size, PartID part_id = 0
) {
    auto metadata = std::make_unique<std::vector<std::uint8_t>>(
        std::vector<std::uint8_t>{1, 2, 3, 4}
    );
    auto data = std::make_unique<rmm::device_buffer>(data_size, stream);
    std::vector<std::uint8_t> host_data;
    if (data_size > 0) {
        host_data.resize(data_size);
        std::iota(host_data.begin(), host_data.end(), std::uint8_t{5});
        RAPIDSMPF_CUDA_TRY(
            cuda_memcpy_async(data->data(), host_data.data(), data_size, stream)
        );
    }
    PackedData packed_data{std::move(metadata), br.move(std::move(data), stream)};
    packed_data.data->latest_write_event().host_wait();
    return Chunk::from_packed_data(1, part_id, std::move(packed_data));
}

std::vector<std::uint8_t> host_bytes(
    std::unique_ptr<Buffer> buffer, cuda::stream_ref stream
) {
    std::vector<std::uint8_t> result(buffer->size);
    RAPIDSMPF_CUDA_TRY(
        cuda_memcpy_async(result.data(), buffer->data(), buffer->size, stream)
    );
    stream.sync();
    return result;
}

}  // namespace

TEST_F(ChunkTest, SpillToHostWhenHostAvailable) {
    constexpr std::size_t data_size = 16;
    auto chunk = make_device_chunk(*br, stream, data_size);
    ASSERT_FALSE(chunk.is_on_disk());
    EXPECT_EQ(chunk.data_memory_type(), MemoryType::DEVICE);

    chunk.spill_from_device(*br);

    ASSERT_FALSE(chunk.is_on_disk());
    EXPECT_TRUE(chunk.is_data_buffer_set());
    EXPECT_NE(chunk.data_memory_type(), MemoryType::DEVICE);
}

TEST_F(ChunkTest, SpillToDiskAndRestore) {
    constexpr std::size_t data_size = 16;
    std::vector<std::uint8_t> expected(data_size);
    std::iota(expected.begin(), expected.end(), std::uint8_t{5});

    auto br_disk = make_disk_spill_buffer_resource();
    auto chunk = make_device_chunk(*br_disk, stream, data_size);
    chunk.spill_from_device(*br_disk);

    EXPECT_TRUE(chunk.is_on_disk());
    EXPECT_FALSE(chunk.is_data_buffer_set());
    EXPECT_FALSE(chunk.is_ready());
    auto const path = chunk.disk_path();
    EXPECT_TRUE(std::filesystem::exists(path));
    EXPECT_THROW(std::ignore = chunk.release_data_buffer(), std::logic_error);
    auto empty = br_disk->move(std::make_unique<rmm::device_buffer>(0, stream), stream);
    EXPECT_THROW(chunk.set_data_buffer(std::move(empty)), std::logic_error);

    chunk.restore_from_disk(*br_disk);
    ASSERT_FALSE(chunk.is_on_disk());
    EXPECT_TRUE(chunk.is_data_buffer_set());
    EXPECT_EQ(chunk.data_memory_type(), MemoryType::DEVICE);
    EXPECT_FALSE(std::filesystem::exists(path));

    auto actual = host_bytes(chunk.release_data_buffer(), stream);
    EXPECT_EQ(actual, expected);
}

TEST_F(ChunkTest, DestroyCleansDiskFile) {
    auto br_disk = make_disk_spill_buffer_resource();
    std::filesystem::path path;
    {
        auto chunk = make_device_chunk(*br_disk, stream, 16);
        chunk.spill_to_disk(*br_disk);
        path = chunk.disk_path();
        EXPECT_TRUE(std::filesystem::exists(path));
    }
    EXPECT_FALSE(std::filesystem::exists(path));
}

TEST(ReceivedChunks, spill_device_to_disk_when_host_full) {
    auto br = make_disk_spill_buffer_resource();
    auto stream = cuda::stream_ref{cudaStreamLegacy};
    ReceivedChunks received;
    constexpr std::size_t chunk_size = 32;

    received.insert(make_device_chunk(*br, stream, chunk_size, /*part_id=*/0));
    EXPECT_EQ(received.spill(br.get(), chunk_size), chunk_size);

    auto chunks = received.extract(0);
    ASSERT_EQ(chunks.size(), 1);
    EXPECT_TRUE(chunks[0].is_on_disk());
    EXPECT_EQ(received.spill(br.get(), chunk_size), 0UL);
}

TEST(ChunksToSend, rejects_disk_resident_chunks) {
    auto br = make_disk_spill_buffer_resource();
    auto stream = cuda::stream_ref{cudaStreamLegacy};
    auto chunk = make_device_chunk(*br, stream, 16);
    chunk.spill_to_disk(*br);
    EXPECT_FALSE(chunk.is_ready());

    ChunksToSend to_send;
    EXPECT_THROW(
        to_send.insert(std::make_unique<Chunk>(std::move(chunk))), std::logic_error
    );
}
