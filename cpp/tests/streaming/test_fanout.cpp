/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <iostream>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cudf_test/table_utilities.hpp>

#include <rapidsmpf/buffer/buffer.hpp>
#include <rapidsmpf/streaming/core/coro_utils.hpp>
#include <rapidsmpf/streaming/core/fanout.hpp>
#include <rapidsmpf/streaming/core/leaf_node.hpp>
#include <rapidsmpf/streaming/core/node.hpp>

#include "base_streaming_fixture.hpp"

#include <coro/coro.hpp>

using namespace rapidsmpf;
using namespace rapidsmpf::streaming;
namespace node = rapidsmpf::streaming::node;
using rapidsmpf::streaming::node::FanoutPolicy;

/**
 * @brief Helper to make a sequence of Message<int> with values [0, n).
 */
std::vector<Message> make_int_inputs(int n) {
    std::vector<Message> inputs;
    inputs.reserve(n);

    Message::CopyCallback copy_cb = [](Message const& msg, MemoryReservation&) {
        return Message{
            msg.sequence_number(),
            std::make_unique<int>(msg.get<int>()),
            ContentDescription{},
            msg.copy_cb()
        };
    };

    for (int i = 0; i < n; ++i) {
        inputs.emplace_back(i, std::make_unique<int>(i), ContentDescription{}, copy_cb);
    }
    return inputs;
}

std::string policy_to_string(FanoutPolicy policy) {
    switch (policy) {
    case FanoutPolicy::BOUNDED:
        return "bounded";
    case FanoutPolicy::UNBOUNDED:
        return "unbounded";
    default:
        return "unknown";
    }
}

class StreamingFanout
    : public BaseStreamingFixture,
      public ::testing::WithParamInterface<std::tuple<FanoutPolicy, int, int, int>> {
  public:
    void SetUp() override {
        std::tie(policy, num_threads, num_out_chs, num_msgs) = GetParam();
        SetUpWithThreads(num_threads);
    }

    FanoutPolicy policy;
    int num_threads;
    int num_out_chs;
    int num_msgs;
};

INSTANTIATE_TEST_SUITE_P(
    StreamingFanout,
    StreamingFanout,
    ::testing::Combine(
        ::testing::Values(FanoutPolicy::BOUNDED, FanoutPolicy::UNBOUNDED),
        ::testing::Values(1, 2, 4),  // number of threads
        ::testing::Values(1, 2, 4),  // number of output channels
        ::testing::Values(1, 10, 100)  // number of messages
    ),
    [](testing::TestParamInfo<StreamingFanout::ParamType> const& info) {
        return "policy_" + policy_to_string(std::get<0>(info.param)) + "_nthreads_"
               + std::to_string(std::get<1>(info.param)) + "_nch_out_"
               + std::to_string(std::get<2>(info.param)) + "_nmsgs_"
               + std::to_string(std::get<3>(info.param));
    }
);

TEST_P(StreamingFanout, SinkPerChannel) {
    // Prepare inputs
    auto inputs = make_int_inputs(num_msgs);

    // Create pipeline
    std::vector<std::vector<Message>> outs(num_out_chs);
    {
        std::vector<Node> nodes;

        auto in = ctx->create_channel();
        nodes.emplace_back(node::push_to_channel(ctx, in, std::move(inputs)));

        std::vector<std::shared_ptr<Channel>> out_chs;
        for (int i = 0; i < num_out_chs; ++i) {
            out_chs.emplace_back(ctx->create_channel());
        }

        nodes.emplace_back(node::fanout(ctx, in, out_chs, policy));

        for (int i = 0; i < num_out_chs; ++i) {
            nodes.emplace_back(node::pull_from_channel(ctx, out_chs[i], outs[i]));
        }

        run_streaming_pipeline(std::move(nodes));
    }

    for (int c = 0; c < num_out_chs; ++c) {
        // Validate sizes
        EXPECT_EQ(outs[c].size(), static_cast<size_t>(num_msgs));

        // Validate ordering/content and that shallow copies share the same underlying
        // object
        for (int i = 0; i < num_msgs; ++i) {
            SCOPED_TRACE("channel " + std::to_string(c) + " idx " + std::to_string(i));
            EXPECT_EQ(outs[c][i].get<int>(), i);
        }
    }
}

enum class ConsumePolicy : uint8_t {
    CHANNEL_ORDER,  // consume all messages from a single channel before moving to the
                    // next
    MESSAGE_ORDER,  // consume messages from all channels before moving to the next
                    // message
};

Node many_input_sink(
    std::shared_ptr<Context> ctx,
    std::vector<std::shared_ptr<Channel>> chs,
    ConsumePolicy consume_policy,
    std::vector<std::vector<Message>>& outs
) {
    ShutdownAtExit c{chs};
    co_await ctx->executor()->schedule();

    if (consume_policy == ConsumePolicy::CHANNEL_ORDER) {
        for (size_t i = 0; i < chs.size(); ++i) {
            while (true) {
                auto msg = co_await chs[i]->receive();
                if (msg.empty()) {
                    break;
                }
                outs[i].push_back(std::move(msg));
            }
        }
    } else if (consume_policy == ConsumePolicy::MESSAGE_ORDER) {
        std::unordered_set<size_t> active_chs{};
        for (size_t i = 0; i < chs.size(); ++i) {
            active_chs.insert(i);
        }
        while (!active_chs.empty()) {
            for (auto it = active_chs.begin(); it != active_chs.end();) {
                auto msg = co_await chs[*it]->receive();
                if (msg.empty()) {
                    it = active_chs.erase(it);
                } else {
                    outs[*it].emplace_back(std::move(msg));
                    it++;
                }
            }
        }
    }
}

struct ManyInputSinkStreamingFanout : public StreamingFanout {
    void run(ConsumePolicy consume_policy) {
        auto inputs = make_int_inputs(num_msgs);

        std::vector<std::vector<Message>> outs(num_out_chs);
        {
            std::vector<Node> nodes;

            auto in = ctx->create_channel();
            nodes.push_back(node::push_to_channel(ctx, in, std::move(inputs)));

            std::vector<std::shared_ptr<Channel>> out_chs;
            for (int i = 0; i < num_out_chs; ++i) {
                out_chs.emplace_back(ctx->create_channel());
            }

            nodes.push_back(node::fanout(ctx, in, out_chs, policy));

            nodes.push_back(many_input_sink(ctx, out_chs, consume_policy, outs));

            run_streaming_pipeline(std::move(nodes));
        }

        std::vector<int> expected(num_msgs);
        std::iota(expected.begin(), expected.end(), 0);
        for (int c = 0; c < num_out_chs; ++c) {
            SCOPED_TRACE("channel " + std::to_string(c));
            std::vector<int> actual;
            actual.reserve(outs[c].size());
            std::ranges::transform(
                outs[c], std::back_inserter(actual), [](const Message& m) {
                    return m.get<int>();
                }
            );
            EXPECT_EQ(expected, actual);
        }
    }
};

INSTANTIATE_TEST_SUITE_P(
    ManyInputSinkStreamingFanout,
    ManyInputSinkStreamingFanout,
    ::testing::Combine(
        ::testing::Values(FanoutPolicy::BOUNDED, FanoutPolicy::UNBOUNDED),
        ::testing::Values(1, 2, 4),  // number of threads
        ::testing::Values(1, 2, 4),  // number of output channels
        ::testing::Values(1, 10, 100)  // number of messages
    ),
    [](testing::TestParamInfo<StreamingFanout::ParamType> const& info) {
        return "policy_" + policy_to_string(std::get<0>(info.param)) + "_nthreads_"
               + std::to_string(std::get<1>(info.param)) + "_nch_out_"
               + std::to_string(std::get<2>(info.param)) + "_nmsgs_"
               + std::to_string(std::get<3>(info.param));
    }
);

TEST_P(ManyInputSinkStreamingFanout, ChannelOrder) {
    if (policy == FanoutPolicy::BOUNDED) {
        GTEST_SKIP() << "Bounded fanout does not support channel order";
    }

    EXPECT_NO_FATAL_FAILURE(run(ConsumePolicy::CHANNEL_ORDER));
}

TEST_P(ManyInputSinkStreamingFanout, MessageOrder) {
    EXPECT_NO_FATAL_FAILURE(run(ConsumePolicy::MESSAGE_ORDER));
}
