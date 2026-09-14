/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <filesystem>
#include <numeric>
#include <stdexcept>

#include <driver_types.h>
#include <gtest/gtest.h>

#include <cuda/stream>

#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/cuda_memcpy_async.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/memory/spill.hpp>
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

    auto data = std::make_unique<rmm::device_buffer>(
        data_size, cuda::stream_ref{cudaStreamLegacy}
    );
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
        std::initializer_list<std::uint8_t>{1, 2, 3, 4}
    );
    auto data = std::make_unique<rmm::device_buffer>(data_size, stream);
    if (data_size > 0) {
        auto const host_data = iota_vector<std::uint8_t>(data_size, 5);
        RAPIDSMPF_CUDA_TRY(
            cuda_memcpy_async(data->data(), host_data.data(), data_size, stream)
        );
    }
    PackedData packed_data{std::move(metadata), br.move(std::move(data), stream)};
    packed_data.data->latest_write_event().host_wait();
    return Chunk::from_packed_data(1, part_id, std::move(packed_data));
}

std::shared_ptr<BufferResource> make_disk_spill_buffer_resource(
    TempDir const& temp_dir, std::int64_t device_limit = 16
) {
    return BufferResource::create(
        rmm::mr::get_current_device_resource_ref(),
        PinnedMemoryDisabled,
        {{MemoryType::DEVICE, device_limit}, {MemoryType::HOST, 0}},
        std::nullopt,
        std::make_shared<StreamPool>(4),
        Statistics::disabled(),
        temp_dir.path()
    );
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

TEST(ReceivedChunks, ReturnsDiskResidentPackedDataForCallerToUnspill) {
    constexpr std::size_t data_size = 32;
    auto const expected = iota_vector<std::uint8_t>(data_size, 5);
    TempDir temp_dir;
    auto br = make_disk_spill_buffer_resource(temp_dir, data_size);
    auto stream = cuda::stream_ref{cudaStreamLegacy};
    ReceivedChunks received;
    received.insert(make_device_chunk(*br, stream, data_size));

    constexpr std::array spillable_memory_types{MemoryType::DISK};
    EXPECT_EQ(received.spill(br.get(), data_size, spillable_memory_types), data_size);

    auto chunks = received.extract(0);
    ASSERT_EQ(chunks.size(), 1);
    auto data = chunks[0].release_data_buffer();
    ASSERT_EQ(data->mem_type(), MemoryType::DISK);
    auto const path = (*data->get_storage<Buffer::DiskBufferT>()).path();
    EXPECT_TRUE(std::filesystem::exists(path));

    std::vector<PackedData> packed_data;
    packed_data.emplace_back(chunks[0].release_metadata_buffer(), std::move(data));
    auto restored =
        unspill_partitions(std::move(packed_data), br.get(), AllowOverbooking::NO);
    ASSERT_EQ(restored.size(), 1);
    EXPECT_EQ(restored[0].data->mem_type(), MemoryType::DEVICE);
    EXPECT_FALSE(std::filesystem::exists(path));
    EXPECT_EQ(host_bytes(std::move(restored[0].data), stream), expected);
}

TEST(ChunksToSend, AcceptsReadyDiskResidentChunks) {
    TempDir temp_dir;
    auto br = make_disk_spill_buffer_resource(temp_dir);
    auto stream = cuda::stream_ref{cudaStreamLegacy};
    auto chunk = make_device_chunk(*br, stream, 16);
    auto reservation = br->reserve_or_fail(chunk.data_size(), MemoryType::DISK);
    chunk.set_data_buffer(br->move(chunk.release_data_buffer(), reservation));
    EXPECT_TRUE(chunk.is_ready());

    ChunksToSend to_send;
    to_send.insert(std::make_unique<Chunk>(std::move(chunk)));
    auto ready = to_send.extract_ready();
    ASSERT_EQ(ready.size(), 1);
    EXPECT_EQ(ready[0].data_memory_type(), MemoryType::DISK);
}
