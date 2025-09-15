/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>
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

    std::vector<Node> nodes;
    auto ch1 = std::make_shared<Channel>();

    // Note, we use a scope to check that coroutines keeps the input alive.
    {
        std::vector<Message> inputs;
        for (int i = 0; i < num_chunks; ++i) {
            inputs.emplace_back(
                std::make_unique<TableChunk>(
                    i,
                    std::make_unique<cudf::table>(
                        expects[i], stream, ctx->br()->device_mr()
                    ),
                    stream
                )
            );
        }

        nodes.push_back(node::push_to_channel(ctx, ch1, std::move(inputs)));
    }

    std::vector<Message> outputs;
    nodes.push_back(node::pull_from_channel(ctx, ch1, outputs));

    run_streaming_pipeline(std::move(nodes));

    EXPECT_EQ(expects.size(), outputs.size());
    for (std::size_t i = 0; i < expects.size(); ++i) {
        EXPECT_EQ(outputs[i].get<TableChunk>().sequence_number(), i);
        CUDF_TEST_EXPECT_TABLES_EQUIVALENT(
            outputs[i].get<TableChunk>().table_view(), expects[i].view()
        );
    }
}

namespace {
Node shutdown(
    std::shared_ptr<Context> ctx,
    std::shared_ptr<ThrottledChannel> ch,
    std::vector<Node>&& tasks
) {
    ShutdownAtExit c{ch};
    auto results = co_await coro::when_all(std::move(tasks));
    for (auto& r : results) {
        r.return_value();
    }
    co_await ch->drain(ctx->executor());
}

Node producer(
    std::shared_ptr<Context> ctx, std::shared_ptr<ThrottledChannel> ch, int val
) {
    co_await ctx->executor()->schedule();
    auto ticket = co_await ch->acquire();
    auto [was_sent, receipt] = co_await ticket.send(Message(std::make_unique<int>(val)));
    RAPIDSMPF_EXPECTS(was_sent, "Channel shut down");
    EXPECT_THROW(
        co_await ticket.send(Message(std::make_unique<int>(val))), std::logic_error
    );
    co_await ctx->executor()->yield();
    co_await receipt.release();
    EXPECT_THROW(co_await receipt.release(), std::logic_error);
}

Node consumer(
    std::shared_ptr<Context> ctx,
    std::shared_ptr<ThrottledChannel> ch,
    std::atomic<int>& result
) {
    ShutdownAtExit c{ch};
    co_await ctx->executor()->schedule();
    while (true) {
        auto msg = co_await ch->receive();
        if (msg.empty()) {
            break;
        }
        auto val = msg.release<int>();
        result.fetch_add(val, std::memory_order_relaxed);
    }
}
}  // namespace

TEST_F(StreamingLeafTasks, ThrottledChannel) {
    auto ch = std::make_shared<ThrottledChannel>(4);
    std::vector<Node> producers;
    std::vector<Node> consumers;
    constexpr int n_producer{100};
    constexpr int n_consumer{3};
    for (int i = 0; i < n_producer; i++) {
        producers.push_back(producer(ctx, ch, i));
    }
    consumers.push_back(shutdown(ctx, ch, std::move(producers)));
    std::atomic<int> result{0};
    for (int i = 0; i < n_consumer; i++) {
        consumers.push_back(consumer(ctx, ch, result));
    }
    run_streaming_pipeline(std::move(consumers));
    EXPECT_EQ(result, ((n_producer - 1) * n_producer) / 2);
}

TEST_F(StreamingLeafTasks, ThrottledChannelThrowInProduce) {
    auto ch = std::make_shared<ThrottledChannel>(1);
    std::vector<Node> producers;
    std::vector<Node> consumers;
    auto make_producer = [](std::shared_ptr<Context> ctx,
                            std::shared_ptr<ThrottledChannel> ch,
                            int i) -> Node {
        co_await producer(ctx, ch, i);
        if (i == 2) {
            throw std::runtime_error("An error occurred");
        }
    };
    constexpr int n_producer{10};
    for (int i = 0; i < n_producer; i++) {
        producers.push_back(make_producer(ctx, ch, i));
    }
    consumers.push_back(shutdown(ctx, ch, std::move(producers)));
    std::atomic<int> result;
    consumers.push_back(consumer(ctx, ch, result));
    EXPECT_THROW(run_streaming_pipeline(std::move(consumers)), std::runtime_error);
}

TEST_F(StreamingLeafTasks, ThrottledChannelThrowInConsume) {
    auto ch = std::make_shared<ThrottledChannel>(1);
    std::vector<Node> producers;
    std::vector<Node> consumers;
    constexpr int n_producer{10};
    for (int i = 0; i < n_producer; i++) {
        producers.push_back(producer(ctx, ch, i));
    }
    consumers.push_back(shutdown(ctx, ch, std::move(producers)));
    auto make_consumer = [](std::shared_ptr<Context> ctx,
                            std::shared_ptr<ThrottledChannel> ch) -> Node {
        ShutdownAtExit c{ch};
        co_await ctx->executor()->yield();
        throw std::runtime_error("An error occurred");
    };
    consumers.push_back(make_consumer(ctx, ch));
    EXPECT_THROW(run_streaming_pipeline(std::move(consumers)), std::runtime_error);
}
