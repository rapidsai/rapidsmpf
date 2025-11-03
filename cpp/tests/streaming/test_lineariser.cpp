/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/leaf_node.hpp>
#include <rapidsmpf/streaming/core/lineariser.hpp>
#include <rapidsmpf/streaming/core/node.hpp>

#include "base_streaming_fixture.hpp"

using namespace rapidsmpf;
using namespace rapidsmpf::streaming;
namespace node = rapidsmpf::streaming::node;

class StreamingLineariser : public BaseStreamingFixture {
    void SetUp() override {
        SetUpWithThreads(8);
    }
};

TEST_F(StreamingLineariser, ManyProducers) {
    constexpr std::size_t num_producers = 10;
    constexpr std::size_t num_messages = 30'000;

    auto ch_out = std::make_shared<Channel>();
    auto lineariser = std::make_shared<Lineariser>(ch_out, num_producers);
    std::vector<Node> tasks;
    tasks.reserve(num_producers + 2);
    auto make_producer = [end = num_messages, stride = num_producers](
                             std::shared_ptr<Context> ctx,
                             std::shared_ptr<Channel> ch_out,
                             std::size_t start
                         ) -> Node {
        for (auto id = start; id < end; id += stride) {
            co_await ctx->executor()->schedule();
            co_await ch_out->send(Message{id, std::make_unique<std::size_t>(id)});
        }
        co_await ch_out->drain(ctx->executor());
    };
    auto inputs = lineariser->get_inputs();
    EXPECT_EQ(inputs.size(), num_producers);
    for (std::size_t i = 0; i < num_producers; i++) {
        tasks.push_back(make_producer(ctx, inputs[i], i));
    }
    tasks.push_back(lineariser->drain(ctx));
    std::vector<Message> outputs;
    outputs.reserve(num_messages);
    tasks.push_back(node::pull_from_channel(ctx, ch_out, outputs));
    run_streaming_pipeline(std::move(tasks));
    EXPECT_EQ(num_messages, outputs.size());
    for (std::size_t i = 0; i < num_messages; i++) {
        EXPECT_EQ(outputs[i].sequence_number(), i);
        EXPECT_EQ(outputs[i].release<std::size_t>(), i);
    }
}
