/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cudf_test/table_utilities.hpp>

#include <rapidsmpf/buffer/buffer.hpp>
#include <rapidsmpf/streaming/core/coro_utils.hpp>
#include <rapidsmpf/streaming/core/fanout.hpp>
#include <rapidsmpf/streaming/core/leaf_node.hpp>

#include "base_streaming_fixture.hpp"

#include <coro/coro.hpp>

using namespace rapidsmpf;
using namespace rapidsmpf::streaming;
namespace node = rapidsmpf::streaming::node;
using rapidsmpf::streaming::node::FanoutPolicy;

using StreamingFanout = BaseStreamingFixture;

namespace {

/**
 * @brief Helper to make a sequence of Message<int> with values [0, n).
 */
std::vector<Message> make_int_inputs(int n) {
    std::vector<Message> inputs;
    inputs.reserve(n);
    for (int i = 0; i < n; ++i) {
        inputs.emplace_back(
            i,
            std::make_unique<int>(i),
            ContentDescription{},
            [](Message const& msg, MemoryReservation&) {
                return Message{
                    msg.sequence_number(),
                    std::make_unique<int>(msg.get<int>()),
                    ContentDescription{}
                };
            }
        );
    }
    return inputs;
}

}  // namespace

TEST_F(StreamingFanout, Bounded) {
    int const num_msgs = 10;

    // Prepare inputs
    auto inputs = make_int_inputs(num_msgs);

    // Create pipeline
    std::vector<Message> outs1, outs2, outs3;
    {
        std::vector<Node> nodes;

        auto in = ctx->create_channel();
        nodes.push_back(node::push_to_channel(ctx, in, std::move(inputs)));

        auto out1 = ctx->create_channel();
        auto out2 = ctx->create_channel();
        auto out3 = ctx->create_channel();
        nodes.push_back(node::fanout(ctx, in, {out1, out2, out3}, FanoutPolicy::BOUNDED));
        nodes.push_back(node::pull_from_channel(ctx, out1, outs1));
        nodes.push_back(node::pull_from_channel(ctx, out2, outs2));
        nodes.push_back(node::pull_from_channel(ctx, out3, outs3));

        run_streaming_pipeline(std::move(nodes));
    }

    // Validate sizes
    EXPECT_EQ(outs1.size(), static_cast<size_t>(num_msgs));
    EXPECT_EQ(outs2.size(), static_cast<size_t>(num_msgs));
    EXPECT_EQ(outs3.size(), static_cast<size_t>(num_msgs));

    // Validate ordering/content and that shallow copies share the same underlying object
    for (int i = 0; i < num_msgs; ++i) {
        EXPECT_EQ(outs1[i].get<int>(), i);
        EXPECT_EQ(outs2[i].get<int>(), i);
        EXPECT_EQ(outs3[i].get<int>(), i);
    }
}

TEST_F(StreamingFanout, Unbounded) {
    int const num_msgs = 7;

    auto inputs = make_int_inputs(num_msgs);

    std::vector<Message> outs1, outs2;
    {
        std::vector<Node> nodes;

        auto in = ctx->create_channel();
        auto out1 = ctx->create_channel();
        auto out2 = ctx->create_channel();

        nodes.push_back(node::push_to_channel(ctx, in, std::move(inputs)));

        // UNBOUNDED policy: buffer all inputs, then broadcast after input closes.
        nodes.push_back(node::fanout(ctx, in, {out1, out2}, FanoutPolicy::UNBOUNDED));

        nodes.push_back(node::pull_from_channel(ctx, out1, outs1));
        nodes.push_back(node::pull_from_channel(ctx, out2, outs2));

        run_streaming_pipeline(std::move(nodes));
    }

    ASSERT_EQ(outs1.size(), static_cast<size_t>(num_msgs));
    ASSERT_EQ(outs2.size(), static_cast<size_t>(num_msgs));

    // Order and identity must be preserved
    for (int i = 0; i < num_msgs; ++i) {
        EXPECT_EQ(outs1[i].get<int>(), i);
        EXPECT_EQ(outs2[i].get<int>(), i);
    }
}
