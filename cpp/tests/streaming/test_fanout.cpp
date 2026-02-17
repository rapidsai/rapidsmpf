/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <iostream>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cudf_test/table_utilities.hpp>

#include <coro/coro.hpp>

#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/pinned_memory_resource.hpp>
#include <rapidsmpf/streaming/core/coro_utils.hpp>
#include <rapidsmpf/streaming/core/fanout.hpp>
#include <rapidsmpf/streaming/core/leaf_node.hpp>
#include <rapidsmpf/streaming/core/node.hpp>

#include "base_streaming_fixture.hpp"

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

/**
 * @brief Helper to make a sequence of Message<Buffer>s where each buffer contains 1024
 * values of int [i, i + 1024).
 */
std::vector<Message> make_buffer_inputs(int n, rapidsmpf::BufferResource& br) {
    std::vector<Message> inputs;
    inputs.reserve(n);

    Message::CopyCallback copy_cb = [&](Message const& msg, MemoryReservation& res) {
        rmm::cuda_stream_view stream = br.stream_pool().get_stream();
        auto const cd = msg.content_description();
        auto buf_cpy = br.allocate(cd.content_size(), stream, res);
        // cd needs to be updated to reflect the new buffer
        ContentDescription new_cd{
            {{buf_cpy->mem_type(), buf_cpy->size}}, ContentDescription::Spillable::YES
        };
        rapidsmpf::buffer_copy(*buf_cpy, msg.get<Buffer>(), cd.content_size());
        return Message{
            msg.sequence_number(), std::move(buf_cpy), std::move(new_cd), msg.copy_cb()
        };
    };
    for (int i = 0; i < n; ++i) {
        std::vector<int> values(1024, 0);
        std::iota(values.begin(), values.end(), i);
        rmm::cuda_stream_view stream = br.stream_pool().get_stream();
        // allocate outside of buffer resource
        auto buffer = br.move(
            std::make_unique<rmm::device_buffer>(
                values.data(), values.size() * sizeof(int), stream
            ),
            stream
        );
        ContentDescription cd{
            std::ranges::single_view{std::pair{MemoryType::DEVICE, 1024 * sizeof(int)}},
            ContentDescription::Spillable::YES
        };
        inputs.emplace_back(i, std::move(buffer), cd, copy_cb);
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

using BaseStreamingFanout = BaseStreamingFixture;

TEST_F(BaseStreamingFanout, InvalidNumberOfOutputChannels) {
    auto in = ctx->create_channel();
    std::vector<std::shared_ptr<Channel>> out_chs;
    out_chs.push_back(ctx->create_channel());
    EXPECT_THROW(
        std::ignore = node::fanout(ctx, in, out_chs, FanoutPolicy::BOUNDED),
        std::invalid_argument
    );
}

class StreamingFanout
    : public BaseStreamingFixture,
      public ::testing::WithParamInterface<std::tuple<FanoutPolicy, int, int, int>> {
  public:
    void SetUp() override {
        std::tie(policy, num_threads, num_out_chs, num_msgs) = GetParam();
        SetUpWithThreads(num_threads);

        // restrict fanout tests to single communicator mode to reduce test runtime
        if (GlobalEnvironment->type() != TestEnvironmentType::SINGLE) {
            GTEST_SKIP() << "Skipping test in non-single communicator mode";
        }
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
        ::testing::Values(1, 4),  // number of threads
        ::testing::Values(2, 4),  // number of output channels
        ::testing::Values(10, 100)  // number of messages
    ),
    [](testing::TestParamInfo<StreamingFanout::ParamType> const& info) {
        return "policy_" + policy_to_string(std::get<0>(info.param)) + "_nthreads_"
               + std::to_string(std::get<1>(info.param)) + "_nch_out_"
               + std::to_string(std::get<2>(info.param)) + "_nmsgs_"
               + std::to_string(std::get<3>(info.param));
    }
);

TEST_P(StreamingFanout, SinkPerChannel) {
    auto inputs = make_int_inputs(num_msgs);

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
            EXPECT_EQ(i, outs[c][i].get<int>());
        }
    }
}

TEST_P(StreamingFanout, SinkPerChannel_Buffer) {
    auto inputs = make_buffer_inputs(num_msgs, *ctx->br());

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
            auto const& buf = outs[c][i].get<Buffer>();
            EXPECT_EQ(1024 * sizeof(int), buf.size);

            std::vector<int> recv(1024);
            buf.stream().synchronize();
            RAPIDSMPF_CUDA_TRY(
                cudaMemcpy(recv.data(), buf.data(), 1024 * sizeof(int), cudaMemcpyDefault)
            );

            EXPECT_TRUE(std::ranges::equal(std::ranges::views::iota(i, i + 1024), recv));
        }
    }
}

namespace {

/**
 * @brief A node that pulls and shuts down a channel after a certain number of messages
 * have been received.
 *
 * @param ctx The context to use.
 * @param ch_in The input channel to receive messages from.
 * @param out_messages The output messages to store the received messages in.
 * @param max_messages The maximum number of messages to receive.
 * @return A coroutine representing the task.
 */
Node shutdown_channel_after_n_messages(
    std::shared_ptr<Context> ctx,
    std::shared_ptr<Channel> ch_in,
    std::vector<Message>& out_messages,
    size_t max_messages
) {
    ShutdownAtExit c{ch_in};
    co_await ctx->executor()->schedule();

    for (size_t i = 0; i < max_messages; ++i) {
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            break;
        }
        out_messages.push_back(std::move(msg));
    }
    co_await ch_in->shutdown();
}

Node throwing_node(std::shared_ptr<Context> ctx, std::shared_ptr<Channel> ch_out) {
    ShutdownAtExit c{ch_out};
    co_await ctx->executor()->schedule();
    throw std::logic_error("throwing source");
}

}  // namespace

// all channels shutsdown after receiving num_msgs / 2 messages
TEST_P(StreamingFanout, SinkPerChannel_ShutdownHalfWay) {
    auto inputs = make_int_inputs(num_msgs);

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
            nodes.emplace_back(
                shutdown_channel_after_n_messages(ctx, out_chs[i], outs[i], num_msgs / 2)
            );
        }

        run_streaming_pipeline(std::move(nodes));
    }

    for (int c = 0; c < num_out_chs; ++c) {
        EXPECT_EQ(static_cast<size_t>(num_msgs / 2), outs[c].size());

        for (int i = 0; i < num_msgs / 2; ++i) {
            SCOPED_TRACE("channel " + std::to_string(c) + " idx " + std::to_string(i));
            EXPECT_EQ(outs[c][i].get<int>(), i);
        }
    }
}

// only odd channels shutdown after receiving num_msgs / 2 messages, others continue to
// receive all messages
TEST_P(StreamingFanout, SinkPerChannel_OddChannelsShutdownHalfWay) {
    auto inputs = make_int_inputs(num_msgs);

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
            if (i % 2 == 0) {
                nodes.emplace_back(node::pull_from_channel(ctx, out_chs[i], outs[i]));
            } else {
                nodes.emplace_back(shutdown_channel_after_n_messages(
                    ctx, out_chs[i], outs[i], num_msgs / 2
                ));
            }
        }

        run_streaming_pipeline(std::move(nodes));
    }

    for (int c = 0; c < num_out_chs; ++c) {
        int expected_size = c % 2 == 0 ? num_msgs : num_msgs / 2;
        EXPECT_EQ(outs[c].size(), expected_size);

        for (int i = 0; i < expected_size; ++i) {
            SCOPED_TRACE("channel " + std::to_string(c) + " idx " + std::to_string(i));
            EXPECT_EQ(outs[c][i].get<int>(), i);
        }
    }
}

class ThrowingStreamingFanout : public StreamingFanout {};

INSTANTIATE_TEST_SUITE_P(
    ThrowingStreamingFanout,
    ThrowingStreamingFanout,
    ::testing::Combine(
        ::testing::Values(FanoutPolicy::BOUNDED, FanoutPolicy::UNBOUNDED),
        ::testing::Values(1, 4),  // number of threads
        ::testing::Values(4),  // number of output channels
        ::testing::Values(10)  // number of messages
    ),
    [](testing::TestParamInfo<StreamingFanout::ParamType> const& info) {
        return "policy_" + policy_to_string(std::get<0>(info.param)) + "_nthreads_"
               + std::to_string(std::get<1>(info.param)) + "_nch_out_"
               + std::to_string(std::get<2>(info.param)) + "_nmsgs_"
               + std::to_string(std::get<3>(info.param));
    }
);

// tests that throwing a source node propagates the error to the pipeline. This test will
// throw, but it should not hang.
TEST_P(ThrowingStreamingFanout, ThrowingSource) {
    std::vector<Node> nodes;

    auto in = ctx->create_channel();
    nodes.emplace_back(throwing_node(ctx, in));

    std::vector<std::shared_ptr<Channel>> out_chs;
    for (int i = 0; i < num_out_chs; ++i) {
        out_chs.emplace_back(ctx->create_channel());
    }

    nodes.emplace_back(node::fanout(ctx, in, out_chs, policy));

    std::vector<Message> dummy_out;
    for (int i = 0; i < num_out_chs; ++i) {
        nodes.emplace_back(node::pull_from_channel(ctx, out_chs[i], dummy_out));
    }

    EXPECT_THROW(run_streaming_pipeline(std::move(nodes)), std::logic_error);
}

// tests that throwing a sink node propagates the error to the pipeline. This test
// will throw, but it should not hang.
TEST_P(ThrowingStreamingFanout, ThrowingSink) {
    auto inputs = make_int_inputs(num_msgs);

    std::vector<Node> nodes;
    auto in = ctx->create_channel();
    nodes.emplace_back(node::push_to_channel(ctx, in, std::move(inputs)));

    std::vector<std::shared_ptr<Channel>> out_chs;
    for (int i = 0; i < num_out_chs; ++i) {
        out_chs.emplace_back(ctx->create_channel());
    }

    nodes.emplace_back(node::fanout(ctx, in, out_chs, policy));

    std::vector<std::vector<Message>> dummy_outs(num_out_chs);
    for (int i = 0; i < num_out_chs; ++i) {
        if (i == 0) {
            nodes.emplace_back(throwing_node(ctx, out_chs[i]));
        } else {
            nodes.emplace_back(node::pull_from_channel(ctx, out_chs[i], dummy_outs[i]));
        }
    }

    EXPECT_THROW(run_streaming_pipeline(std::move(nodes)), std::logic_error);
}

namespace {
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
}  // namespace

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
        ::testing::Values(1, 4),  // number of threads
        ::testing::Values(2, 4),  // number of output channels
        ::testing::Values(10, 100)  // number of messages
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

class SpillingStreamingFanout : public BaseStreamingFixture {
    void SetUp() override {
        SetUpWithThreads(4);

        // override br and context with no device memory
        std::unordered_map<MemoryType, BufferResource::MemoryAvailable> memory_available =
            {
                {MemoryType::DEVICE, []() -> std::int64_t { return 0; }},
            };
        br = std::make_shared<rapidsmpf::BufferResource>(
            mr_cuda, rapidsmpf::PinnedMemoryResource::Disabled, memory_available
        );
        auto options = ctx->options();
        ctx = std::make_shared<rapidsmpf::streaming::Context>(
            options, GlobalEnvironment->comm_, br
        );
    }
};

TEST_F(SpillingStreamingFanout, Spilling) {
    auto inputs = make_buffer_inputs(100, *ctx->br());
    constexpr int num_out_chs = 4;
    constexpr FanoutPolicy policy = FanoutPolicy::UNBOUNDED;

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
        nodes.push_back(
            many_input_sink(ctx, out_chs, ConsumePolicy::CHANNEL_ORDER, outs)
        );

        run_streaming_pipeline(std::move(nodes));
    }

    for (int c = 0; c < num_out_chs; ++c) {
        SCOPED_TRACE("channel " + std::to_string(c));
        // all messages should be in host memory
        EXPECT_TRUE(std::ranges::all_of(outs[c], [](const Message& m) {
            auto const cd = m.content_description();
            return cd.principal_memory_type() == MemoryType::HOST
                   && cd.content_size() == 1024 * sizeof(int);
        }));
    }
}
