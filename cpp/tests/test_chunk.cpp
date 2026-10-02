/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <driver_types.h>
#include <gtest/gtest.h>

#include <cuda/stream>

#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
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

/// Creates a chunk whose device data is written on @p stream (no host wait).
std::unique_ptr<Chunk> make_chunk_on_stream(
    BufferResource& br,
    cuda::stream_ref stream,
    ChunkID chunk_id,
    PartID part_id,
    std::size_t data_size = 16
) {
    auto metadata = std::make_unique<std::vector<std::uint8_t>>(
        std::initializer_list<std::uint8_t>{1, 2, 3, 4}
    );
    auto data = std::make_unique<rmm::device_buffer>(data_size, stream);
    PackedData packed_data{std::move(metadata), br.move(std::move(data), stream)};
    return std::make_unique<Chunk>(
        Chunk::from_packed_data(chunk_id, part_id, std::move(packed_data))
    );
}

std::vector<ChunkID> chunk_ids(std::vector<Chunk> const& chunks) {
    std::vector<ChunkID> ret;
    for (auto const& chunk : chunks) {
        ret.push_back(chunk.chunk_id());
    }
    return ret;
}

}  // namespace

TEST(SendOrderPolicy, Parse) {
    EXPECT_EQ(parse_send_order_policy("global"), SendOrderPolicy::Global);
    EXPECT_EQ(parse_send_order_policy(" Rank "), SendOrderPolicy::PerRank);
    EXPECT_EQ(parse_send_order_policy("PID"), SendOrderPolicy::PerPartition);
    EXPECT_EQ(parse_send_order_policy("none"), SendOrderPolicy::None);
    EXPECT_THROW(std::ignore = parse_send_order_policy("fifo"), std::invalid_argument);
}

class ChunksToSendPolicyTest : public ::testing::TestWithParam<SendOrderPolicy> {};

INSTANTIATE_TEST_SUITE_P(
    ChunksToSend,
    ChunksToSendPolicyTest,
    ::testing::Values(
        SendOrderPolicy::Global,
        SendOrderPolicy::PerRank,
        SendOrderPolicy::PerPartition,
        SendOrderPolicy::None
    ),
    [](auto const& info) {
        std::stringstream ss;
        ss << info.param;
        return ss.str();
    }
);

TEST_P(ChunksToSendPolicyTest, NotReadyChunkBlocksPerPolicy) {
    auto const policy = GetParam();
    auto br = BufferResource::create(rmm::mr::get_current_device_resource_ref());
    auto const ready_stream = cuda::stream_ref{cudaStreamLegacy};
    StreamGate gate;

    // {chunk ID, destination rank, partition ID}; chunk 0 is not ready.
    ChunksToSend to_send{policy};
    to_send.insert(0, make_chunk_on_stream(*br, gate.stream(), 0, 0));
    to_send.insert(0, make_chunk_on_stream(*br, ready_stream, 1, 0));
    to_send.insert(0, make_chunk_on_stream(*br, ready_stream, 2, 2));
    to_send.insert(1, make_chunk_on_stream(*br, ready_stream, 3, 1));
    to_send.insert(1, make_chunk_on_stream(*br, ready_stream, 4, 1));
    ready_stream.sync();

    std::vector<ChunkID> expected_first;
    std::vector<ChunkID> expected_second;
    std::size_t expected_blocked = 0;
    switch (policy) {
    case SendOrderPolicy::Global:
        expected_first = {};
        expected_second = {0, 1, 2, 3, 4};
        expected_blocked = 4;
        break;
    case SendOrderPolicy::PerRank:
        expected_first = {3, 4};
        expected_second = {0, 1, 2};
        expected_blocked = 2;
        break;
    case SendOrderPolicy::PerPartition:
        expected_first = {2, 3, 4};
        expected_second = {0, 1};
        expected_blocked = 1;
        break;
    case SendOrderPolicy::None:
        expected_first = {1, 2, 3, 4};
        expected_second = {0};
        expected_blocked = 0;
        break;
    }

    ChunksToSend::ExtractStats stats;
    // Keep the extracted chunks alive until the gate is open: freeing device memory
    // (cudaFree) synchronizes the device and would wait on the gated stream.
    auto const first = to_send.extract_ready(&stats);
    EXPECT_EQ(chunk_ids(first), expected_first);
    EXPECT_EQ(stats.not_ready, 1);
    EXPECT_EQ(stats.blocked, expected_blocked);
    EXPECT_FALSE(to_send.empty());

    gate.open();
    gate.stream().sync();
    EXPECT_EQ(chunk_ids(to_send.extract_ready(&stats)), expected_second);
    EXPECT_EQ(stats.not_ready, 0);
    EXPECT_EQ(stats.blocked, 0);
    EXPECT_TRUE(to_send.empty());
}
