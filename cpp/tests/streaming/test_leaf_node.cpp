/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cudf_test/table_utilities.hpp>

#include <rapidsmpf/buffer/buffer.hpp>
#include <rapidsmpf/communicator/single.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/leaf_node.hpp>
#include <rapidsmpf/streaming/core/node.hpp>
#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

#include "../utils.hpp"
#include "base_streaming_fixture.hpp"

using namespace rapidsmpf;
using namespace rapidsmpf::streaming;
namespace node = rapidsmpf::streaming::node;

using StreamingLeafTasks = BaseStreamingFixture;

TEST_F(StreamingLeafTasks, PushAndPullChunks) {
    constexpr int num_rows = 100;
    constexpr int num_chunks = 10;

    std::vector<cudf::table> expects;
    for (int i = 0; i < num_chunks; ++i) {
        expects.emplace_back(random_table_with_index(i, num_rows, 0, 10));
    }

    std::vector<std::unique_ptr<TableChunk>> inputs;
    for (int i = 0; i < num_chunks; ++i) {
        inputs.emplace_back(
            std::make_unique<TableChunk>(
                i,
                std::make_unique<cudf::table>(
                    expects[i], ctx->stream(), ctx->br()->device_mr()
                ),
                ctx->stream()
            )
        );
    }

    std::vector<Node> nodes;
    auto ch1 = make_shared_channel<TableChunk>();
    nodes.push_back(
        node::push_chunks_to_channel<TableChunk>(ctx, ch1, std::move(inputs))
    );

    std::vector<std::unique_ptr<TableChunk>> outputs;
    nodes.push_back(node::pull_chunks_from_channel(ctx, ch1, outputs));

    run_streaming_pipeline(std::move(nodes));

    EXPECT_EQ(expects.size(), outputs.size());
    for (std::size_t i = 0; i < expects.size(); ++i) {
        EXPECT_EQ(outputs[i]->sequence_number(), i);
        CUDF_TEST_EXPECT_TABLES_EQUIVALENT(outputs[i]->table_view(), expects[i].view());
    }
}
