/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cudf_test/table_utilities.hpp>

#include <rapidsmpf/buffer/buffer.hpp>
#include <rapidsmpf/communicator/single.hpp>
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
            std::make_unique<TableChunk>(
                i,
                std::make_unique<cudf::table>(expects[i], stream, ctx->br()->device_mr()),
                stream
            )
        );
    }

    // Create and run the streaming pipeline.
    std::vector<Message> outputs;
    {
        std::vector<Node> nodes;
        auto ch1 = std::make_shared<Channel2>();
        nodes.push_back(node::push_to_channel(ctx, ch1, std::move(inputs)));

        auto ch2 = std::make_shared<Channel2>();
        nodes.push_back(
            node::partition_and_pack2(
                ctx, ch1, ch2, {1}, num_partitions, hash_function, seed
            )
        );

        auto ch3 = std::make_shared<Channel2>();
        nodes.push_back(node::unpack_and_concat2(ctx, ch2, ch3));

        nodes.push_back(node::pull_from_channel(ctx, ch3, outputs));

        run_streaming_pipeline(std::move(nodes));
    }

    EXPECT_EQ(expects.size(), outputs.size());
    for (std::size_t i = 0; i < expects.size(); ++i) {
        auto output = outputs[i].release<TableChunk>();
        EXPECT_EQ(output->sequence_number(), i);
        CUDF_TEST_EXPECT_TABLES_EQUIVALENT(
            sort_table(output->table_view()), sort_table(expects[i].view())
        );
    }
}
