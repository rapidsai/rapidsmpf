/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/node.hpp>

#include "base_streaming_fixture.hpp"

using namespace rapidsmpf;
using namespace rapidsmpf::streaming;

using StreamingChannel = BaseStreamingFixture;

TEST_F(StreamingChannel, ReceiveMessageId) {
    constexpr int num_messages = 10;
    auto ch = ctx->create_channel();
    std::vector<Node> nodes;

    nodes.push_back([](auto ctx, auto ch_out) -> Node {
        ShutdownAtExit c{ch_out};
        co_await ctx->executor()->schedule();
        for (int i = 0; i < num_messages; ++i) {
            co_await ch_out->send(Message{
                static_cast<uint64_t>(i),
                std::make_unique<int>(i * 10),
                ContentDescription{},
                [](Message const& /* msg */, MemoryReservation& /* res */) -> Message {
                    RAPIDSMPF_FAIL("should not be called");
                }
            });
        }
        co_await ch_out->drain(ctx->executor());
    }(ctx, ch));

    std::vector<int> recv_vals;
    std::vector<uint64_t> recv_seq_nums;
    nodes.push_back(
        [](auto ctx, auto ch_in, std::vector<int>& values, std::vector<uint64_t>& seq_nums
        ) -> Node {
            ShutdownAtExit c{ch_in};
            co_await ctx->executor()->schedule();
            while (true) {
                auto msg_id = co_await ch_in->receive_message_id();
                if (msg_id == SpillableMessages::InvalidMessageId) {
                    break;
                }
                auto msg = ctx->spillable_messages()->extract(msg_id);
                seq_nums.push_back(msg.sequence_number());
                values.push_back(msg.template get<int>());
            }
        }(ctx, ch, recv_vals, recv_seq_nums)
    );

    run_streaming_pipeline(std::move(nodes));

    // Verify all messages were received correctly.
    EXPECT_EQ(num_messages, recv_vals.size());
    EXPECT_EQ(num_messages, recv_seq_nums.size());

    std::ranges::sort(recv_seq_nums);
    std::ranges::sort(recv_vals);
    for (int i = 0; i < num_messages; ++i) {
        EXPECT_EQ(static_cast<uint64_t>(i), recv_seq_nums[i]);
        EXPECT_EQ(i * 10, recv_vals[i]);
    }
}
