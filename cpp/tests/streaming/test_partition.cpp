/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gmock/gmock.h>

#include <rapidsmpf/communicator/single.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/shuffler/chunk.hpp>
#include <rapidsmpf/streaming/chunks/partition.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/leaf_actor.hpp>

#include "../utils.hpp"
#include "base_streaming_fixture.hpp"

using StreamingPartition = BaseStreamingFixture;

TEST_F(StreamingPartition, PartitionMapChunkToMessage) {
    constexpr std::uint64_t seq = 42;
    std::unordered_map<rapidsmpf::shuffler::PartID, rapidsmpf::PackedData> data;
    data.emplace(0, generate_packed_data(10, 0, stream, *br));
    data.emplace(1, generate_packed_data(10, 10, stream, *br));
    auto chunk =
        std::make_unique<rapidsmpf::streaming::PartitionMapChunk>(std::move(data));

    rapidsmpf::streaming::Message m = to_message(seq, std::move(chunk));
    EXPECT_FALSE(m.empty());
    EXPECT_TRUE(m.holds<rapidsmpf::streaming::PartitionMapChunk>());
    EXPECT_TRUE(m.content_description().spillable());
    EXPECT_EQ(m.content_description().content_size(rapidsmpf::MemoryType::HOST), 0);
    EXPECT_EQ(m.content_description().content_size(rapidsmpf::MemoryType::DEVICE), 80);
    EXPECT_EQ(m.sequence_number(), seq);

    auto res = br->reserve_or_fail(m.copy_cost(), rapidsmpf::MemoryType::DEVICE);
    rapidsmpf::streaming::Message m2 = m.copy(res);
    EXPECT_EQ(res.size(), 0);
    EXPECT_FALSE(m2.empty());
    EXPECT_TRUE(m2.holds<rapidsmpf::streaming::PartitionMapChunk>());
    EXPECT_TRUE(m2.content_description().spillable());
    EXPECT_EQ(m2.content_description().content_size(rapidsmpf::MemoryType::HOST), 0);
    EXPECT_EQ(m2.content_description().content_size(rapidsmpf::MemoryType::DEVICE), 80);

    auto chunk2 = m2.release<rapidsmpf::streaming::PartitionMapChunk>();
    validate_packed_data(std::move(chunk2.data.at(0)), 10, 0, stream, *br);
    validate_packed_data(std::move(chunk2.data.at(1)), 10, 10, stream, *br);
}

TEST_F(StreamingPartition, PartitionMapChunkContentDescription) {
    // Create a packed data, one in device and one in host memory.
    std::unordered_map<rapidsmpf::shuffler::PartID, rapidsmpf::PackedData> data;
    auto pack1 = generate_packed_data(5, 0, stream, *br);
    auto pack1_size = pack1.data->size;
    auto pack2_size = pack1.data->size * 2;
    auto res = br->reserve_or_fail(pack2_size, rapidsmpf::MemoryType::HOST);
    auto pack2 = generate_packed_data(10, 0, stream, *br).copy(res);
    data.emplace(0, std::move(pack1));
    data.emplace(1, std::move(pack2));

    auto chunk =
        std::make_unique<rapidsmpf::streaming::PartitionMapChunk>(std::move(data));
    auto cd = get_content_description(*chunk);
    EXPECT_TRUE(cd.spillable());
    EXPECT_EQ(cd.content_size(rapidsmpf::MemoryType::DEVICE), pack1_size);
    EXPECT_EQ(cd.content_size(rapidsmpf::MemoryType::HOST), pack2_size);
}

TEST_F(StreamingPartition, PartitionVectorChunkContentDescription) {
    // Create a packed data, one in device and one in host memory.
    std::vector<rapidsmpf::PackedData> data;
    auto pack1 = generate_packed_data(5, 0, stream, *br);
    auto pack1_size = pack1.data->size;
    auto pack2_size = pack1.data->size * 2;
    auto res = br->reserve_or_fail(pack2_size, rapidsmpf::MemoryType::HOST);
    auto pack2 = generate_packed_data(10, 0, stream, *br).copy(res);
    data.push_back(std::move(pack1));
    data.push_back(std::move(pack2));

    auto chunk =
        std::make_unique<rapidsmpf::streaming::PartitionVectorChunk>(std::move(data));
    auto cd = get_content_description(*chunk);
    EXPECT_TRUE(cd.spillable());
    EXPECT_EQ(cd.content_size(rapidsmpf::MemoryType::DEVICE), pack1_size);
    EXPECT_EQ(cd.content_size(rapidsmpf::MemoryType::HOST), pack2_size);
}
