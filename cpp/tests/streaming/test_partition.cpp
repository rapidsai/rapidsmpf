/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cudf_test/table_utilities.hpp>

#include <rapidsmpf/buffer/buffer.hpp>
#include <rapidsmpf/communicator/single.hpp>
#include <rapidsmpf/shuffler/chunk.hpp>
#include <rapidsmpf/streaming/chunks/partition.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/leaf_node.hpp>
#include <rapidsmpf/streaming/cudf/partition.hpp>
#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

#include "../utils.hpp"
#include "base_streaming_fixture.hpp"

using namespace rapidsmpf;
using namespace rapidsmpf::streaming;
namespace node = rapidsmpf::streaming::node;

using StreamingPartition = BaseStreamingFixture;

TEST_F(StreamingPartition, PackUnpackRoundTrip) {
    int const num_partitions = 5;
    int const num_rows = 100;
    int const num_chunks = 10;
    std::int64_t const seed = 42;
    constexpr cudf::hash_id hash_function = cudf::hash_id::HASH_MURMUR3;

    std::vector<cudf::table> expects;
    for (int i = 0; i < num_chunks; ++i) {
        expects.push_back(random_table_with_index(seed + i, num_rows, 0, 10));
    }

    std::vector<Message> inputs;
    for (int i = 0; i < num_chunks; ++i) {
        inputs.emplace_back(
            i,
            std::make_unique<TableChunk>(
                std::make_unique<cudf::table>(expects[i], stream, ctx->br()->device_mr()),
                stream
            )
        );
    }

    // Create and run the streaming pipeline.
    std::vector<Message> outputs;
    {
        std::vector<Node> nodes;
        auto ch1 = std::make_shared<Channel>();
        nodes.push_back(node::push_to_channel(ctx, ch1, std::move(inputs)));

        auto ch2 = std::make_shared<Channel>();
        nodes.push_back(
            node::partition_and_pack(
                ctx, ch1, ch2, {1}, num_partitions, hash_function, seed
            )
        );

        auto ch3 = std::make_shared<Channel>();
        nodes.push_back(node::unpack_and_concat(ctx, ch2, ch3));

        nodes.push_back(node::pull_from_channel(ctx, ch3, outputs));

        run_streaming_pipeline(std::move(nodes));
    }

    EXPECT_EQ(expects.size(), outputs.size());
    for (std::size_t i = 0; i < expects.size(); ++i) {
        EXPECT_EQ(outputs[i].sequence_number(), i);
        auto output = outputs[i].release<TableChunk>();
        CUDF_TEST_EXPECT_TABLES_EQUIVALENT(
            sort_table(output.table_view()), sort_table(expects[i].view())
        );
    }
}

TEST_F(StreamingPartition, PartitionMapChunkToMessage) {
    constexpr std::uint64_t seq = 42;
    std::unordered_map<shuffler::PartID, PackedData> data;
    data.emplace(0, generate_packed_data(10, 0, stream, *br));
    data.emplace(1, generate_packed_data(10, 10, stream, *br));
    PartitionMapChunk chunk{std::move(data)};

    Message m = to_message(seq, std::move(chunk));
    EXPECT_FALSE(m.empty());
    EXPECT_TRUE(m.holds<PartitionMapChunk>());
    EXPECT_EQ(m.primary_data_size(MemoryType::HOST), std::make_pair(0, true));
    EXPECT_EQ(m.primary_data_size(MemoryType::DEVICE), std::make_pair(80, true));
    EXPECT_EQ(m.sequence_number(), seq);

    auto res = br->reserve_or_fail(
        m.primary_data_size(MemoryType::DEVICE).first, MemoryType::DEVICE
    );
    Message m2 = m.copy(br.get(), res);
    EXPECT_EQ(res.size(), 0);
    EXPECT_FALSE(m2.empty());
    EXPECT_TRUE(m2.holds<PartitionMapChunk>());
    EXPECT_EQ(m2.primary_data_size(MemoryType::HOST), std::make_pair(0, true));
    EXPECT_EQ(m2.primary_data_size(MemoryType::DEVICE), std::make_pair(80, true));

    auto chunk2 = m2.release<PartitionMapChunk>();
    validate_packed_data(std::move(chunk2.data.at(0)), 10, 0, stream, *br);
    validate_packed_data(std::move(chunk2.data.at(1)), 10, 10, stream, *br);
}
