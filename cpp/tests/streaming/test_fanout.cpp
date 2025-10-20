/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cudf_test/table_utilities.hpp>

#include <rapidsmpf/buffer/buffer.hpp>
#include <rapidsmpf/streaming/core/fanout.hpp>
#include <rapidsmpf/streaming/core/leaf_node.hpp>

#include "base_streaming_fixture.hpp"

using namespace rapidsmpf;
using namespace rapidsmpf::streaming;
namespace node = rapidsmpf::streaming::node;
using rapidsmpf::streaming::node::FanoutPolicy;

using StreamingBCast = BaseStreamingFixture;

namespace {

/**
 * @brief Helper to make a sequence of Message<int> with values [0, n).
 */
std::vector<Message> make_int_inputs(int n) {
    std::vector<Message> inputs;
    inputs.reserve(n);
    for (int i = 0; i < n; ++i) {
        inputs.emplace_back(std::make_unique<int>(i));
    }
    return inputs;
}

}  // namespace

TEST_F(StreamingBCast, BoundedReplicates) {
    int const num_msgs = 10;

    // Prepare inputs
    auto inputs = make_int_inputs(num_msgs);

    // Create pipeline
    std::vector<Message> outs1, outs2, outs3;
    {
        std::vector<Node> nodes;

        auto in = std::make_shared<Channel>();
        nodes.push_back(node::push_to_channel(ctx, in, std::move(inputs)));

        auto out1 = std::make_shared<Channel>();
        auto out2 = std::make_shared<Channel>();
        auto out3 = std::make_shared<Channel>();
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
        // Same address means same shared payload (shallow copy)
        EXPECT_EQ(
            std::addressof(outs1[i].get<int>()), std::addressof(outs2[i].get<int>())
        );
        EXPECT_EQ(
            std::addressof(outs1[i].get<int>()), std::addressof(outs3[i].get<int>())
        );
        EXPECT_EQ(outs1[i].get<int>(), i);
        EXPECT_EQ(outs2[i].get<int>(), i);
        EXPECT_EQ(outs3[i].get<int>(), i);
    }

    // release() semantics: requires sole ownership.
    // For each triplet, drop two references, then release from the remaining one.
    for (int i = 0; i < num_msgs; ++i) {
        // Holding 3 references -> release must fail
        EXPECT_THROW(std::ignore = outs1[i].release<int>(), std::invalid_argument);

        // Make outs1[i] the sole owner
        outs2[i].reset();
        outs3[i].reset();

        // Now release succeeds and yields the expected value
        EXPECT_NO_THROW({
            int v = outs1[i].release<int>();
            EXPECT_EQ(v, i);
        });

        // After release, the message is empty
        EXPECT_TRUE(outs1[i].empty());
    }
}

TEST_F(StreamingBCast, UnboundedReplicates) {
    int const num_msgs = 7;

    auto inputs = make_int_inputs(num_msgs);

    std::vector<Message> outs1, outs2;
    {
        std::vector<Node> nodes;

        auto in = std::make_shared<Channel>();
        auto out1 = std::make_shared<Channel>();
        auto out2 = std::make_shared<Channel>();

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
        EXPECT_EQ(&outs1[i].get<int>(), &outs2[i].get<int>());
        EXPECT_EQ(outs1[i].get<int>(), i);
        EXPECT_EQ(outs2[i].get<int>(), i);
    }

    // Release semantics: with two refs, release must fail until one is reset.
    for (int i = 0; i < num_msgs; ++i) {
        EXPECT_THROW(std::ignore = outs1[i].release<int>(), std::invalid_argument);

        // Make outs1[i] sole owner
        outs2[i].reset();

        EXPECT_NO_THROW({
            int v = outs1[i].release<int>();
            EXPECT_EQ(v, i);
        });
        EXPECT_TRUE(outs1[i].empty());
    }
}
