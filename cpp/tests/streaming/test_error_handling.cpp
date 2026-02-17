/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/node.hpp>

#include "base_streaming_fixture.hpp"

using namespace rapidsmpf;
using namespace rapidsmpf::streaming;

using StreamingErrorHandling = BaseStreamingFixture;

TEST_F(StreamingErrorHandling, UnhandledException) {
    std::vector<Actor> nodes;

    nodes.push_back([](Context& ctx) -> Actor {
        co_await ctx.executor()->schedule();
        throw std::runtime_error("unhandled_exception");
    }(*ctx));

    EXPECT_THROW(run_actor_graph(std::move(nodes)), std::runtime_error);
}

TEST_F(StreamingErrorHandling, ProducerThrows) {
    auto ch = ctx->create_channel();
    std::vector<Actor> nodes;

    // Producer node.
    nodes.push_back(
        [](std::shared_ptr<Context> ctx, std::shared_ptr<Channel> ch_out) -> Actor {
            ShutdownAtExit c{ch_out};
            co_await ctx->executor()->schedule();
            throw std::runtime_error("some unhandled exception");
        }(ctx, ch)
    );

    // Consumer node.
    nodes.push_back(
        [](std::shared_ptr<Context> ctx, std::shared_ptr<Channel> ch_in) -> Actor {
            ShutdownAtExit c{ch_in};
            co_await ctx->executor()->schedule();
            std::ignore = ch_in->receive();
        }(ctx, ch)
    );

    EXPECT_THROW(run_actor_graph(std::move(nodes)), std::runtime_error);
}

TEST_F(StreamingErrorHandling, ConsumerThrows) {
    auto ch = ctx->create_channel();
    std::vector<Actor> nodes;

    // Producer node.
    nodes.push_back(
        [](std::shared_ptr<Context> ctx, std::shared_ptr<Channel> ch_out) -> Actor {
            ShutdownAtExit c{ch_out};
            co_await ctx->executor()->schedule();
            co_await ch_out->send(
                Message{0, std::make_unique<int>(42), ContentDescription{}}
            );
            co_await ch_out->drain(ctx->executor());
        }(ctx, ch)
    );

    // Consumer node.
    nodes.push_back(
        [](std::shared_ptr<Context> ctx, std::shared_ptr<Channel> ch_in) -> Actor {
            ShutdownAtExit c{ch_in};
            co_await ctx->executor()->schedule();
            throw std::runtime_error("some unhandled exception");
        }(ctx, ch)
    );

    EXPECT_THROW(run_actor_graph(std::move(nodes)), std::runtime_error);
}
