/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rapidsmpf/memory/content_description.hpp>
#include <rapidsmpf/streaming/core/actor.hpp>
#include <rapidsmpf/streaming/core/coro_utils.hpp>
#include <rapidsmpf/streaming/core/leaf_actor.hpp>
#include <rapidsmpf/streaming/core/message.hpp>

#include "base_streaming_fixture.hpp"

using namespace rapidsmpf::streaming;

namespace {
std::vector<Message> make_int_messages(std::size_t n) {
    std::vector<Message> messages;
    messages.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        messages.emplace_back(
            i, std::make_unique<int>(i), rapidsmpf::ContentDescription{}
        );
    }
    return messages;
}
}  // namespace

using StreamingChannel = BaseStreamingFixture;

TEST_F(StreamingChannel, DataRoundTripWithoutMetadata) {
    auto ch = ctx->create_channel();
    std::vector<Message> outputs;
    std::vector<Actor> actors;
    static constexpr std::size_t num_messages = 4;
    actors.emplace_back(actor::push_to_channel(ctx, ch, make_int_messages(num_messages)));
    actors.emplace_back(actor::pull_from_channel(ctx, ch, outputs));
    run_actor_graph(std::move(actors));

    ASSERT_EQ(outputs.size(), num_messages);
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(outputs[static_cast<std::size_t>(i)].release<int>(), i);
    }
}

TEST_F(StreamingChannel, MetadataSendReceiveAndShutdown) {
    auto ch = ctx->create_channel();
    std::vector<Message> metadata_outputs;
    std::vector<Message> data_outputs;
    std::vector<Actor> actors;

    auto producer = [this, ch]() -> Actor {
        ShutdownAtExit c{ch};
        co_await ctx->executor()->schedule();

        auto meta_task = [&]() -> Actor {
            co_await ch->send_metadata(Message{0, std::make_unique<int>(10), {}});
            co_await ch->send_metadata(Message{1, std::make_unique<int>(20), {}});
            co_await ch->drain_metadata(ctx->executor());
        };
        auto send_task = [&]() -> Actor {
            co_await ch->send(Message{0, std::make_unique<int>(1), {}});
            co_await ch->send(Message{1, std::make_unique<int>(2), {}});
            co_await ch->drain(ctx->executor());
        };
        coro_results(co_await coro::when_all(meta_task(), send_task()));
    };

    auto consumer = [this, ch, &metadata_outputs, &data_outputs]() -> Actor {
        ShutdownAtExit c{ch};
        co_await ctx->executor()->schedule();

        while (true) {
            auto msg = co_await ch->receive_metadata();
            if (msg.empty()) {
                break;
            }
            metadata_outputs.push_back(std::move(msg));
        }

        while (true) {
            auto msg = co_await ch->receive();
            if (msg.empty()) {
                break;
            }
            data_outputs.push_back(std::move(msg));
        }
    };

    actors.emplace_back(producer());
    actors.emplace_back(consumer());
    run_actor_graph(std::move(actors));

    ASSERT_EQ(metadata_outputs.size(), 2U);
    EXPECT_EQ(metadata_outputs[0].get<int>(), 10);
    EXPECT_EQ(metadata_outputs[1].get<int>(), 20);

    ASSERT_EQ(data_outputs.size(), 2U);
    EXPECT_EQ(data_outputs[0].get<int>(), 1);
    EXPECT_EQ(data_outputs[1].get<int>(), 2);
}

TEST_F(StreamingChannel, DataOnlyWithMetadataShutdown) {
    auto ch = ctx->create_channel();
    std::vector<Message> data_outputs;
    std::vector<Message> metadata_outputs;
    std::vector<Actor> actors;

    auto producer = [this, ch]() -> Actor {
        ShutdownAtExit c{ch};
        co_await ctx->executor()->schedule();

        co_await ch->shutdown_metadata();
        co_await ch->send(
            Message{0, std::make_unique<int>(10), rapidsmpf::ContentDescription{}}
        );
        co_await ch->send(
            Message{1, std::make_unique<int>(20), rapidsmpf::ContentDescription{}}
        );
        co_await ch->drain(ctx->executor());
    };

    auto consumer = [this, ch, &metadata_outputs, &data_outputs]() -> Actor {
        ShutdownAtExit c{ch};
        co_await ctx->executor()->schedule();

        while (true) {
            auto msg = co_await ch->receive_metadata();
            if (msg.empty()) {
                break;
            }
            metadata_outputs.push_back(std::move(msg));
        }

        while (true) {
            auto msg = co_await ch->receive();
            if (msg.empty()) {
                break;
            }
            data_outputs.push_back(std::move(msg));
        }
    };

    actors.emplace_back(producer());
    actors.emplace_back(consumer());
    run_actor_graph(std::move(actors));

    EXPECT_TRUE(metadata_outputs.empty());
    ASSERT_EQ(data_outputs.size(), 2U);
    EXPECT_EQ(data_outputs[0].get<int>(), 10);
    EXPECT_EQ(data_outputs[1].get<int>(), 20);
}

TEST_F(StreamingChannel, MetadataOnlyWithDataShutdown) {
    auto ch = ctx->create_channel();
    std::vector<Message> metadata_outputs;
    std::vector<Message> data_outputs;
    std::vector<Actor> actors;

    auto producer = [this, ch]() -> Actor {
        ShutdownAtExit c{ch};
        co_await ctx->executor()->schedule();

        co_await ch->send_metadata(
            Message{0, std::make_unique<int>(10), rapidsmpf::ContentDescription{}}
        );
        co_await ch->send_metadata(
            Message{1, std::make_unique<int>(20), rapidsmpf::ContentDescription{}}
        );
        co_await ch->drain(ctx->executor());
    };

    auto consumer = [this, ch, &metadata_outputs, &data_outputs]() -> Actor {
        ShutdownAtExit c{ch};
        co_await ctx->executor()->schedule();

        while (true) {
            auto msg = co_await ch->receive_metadata();
            if (msg.empty()) {
                break;
            }
            metadata_outputs.push_back(std::move(msg));
        }

        while (true) {
            auto msg = co_await ch->receive();
            if (msg.empty()) {
                break;
            }
            data_outputs.push_back(std::move(msg));
        }
    };

    actors.emplace_back(producer());
    actors.emplace_back(consumer());
    run_actor_graph(std::move(actors));

    ASSERT_EQ(metadata_outputs.size(), 2U);
    EXPECT_EQ(metadata_outputs[0].get<int>(), 10);
    EXPECT_EQ(metadata_outputs[1].get<int>(), 20);
    EXPECT_TRUE(data_outputs.empty());
}

TEST_F(StreamingChannel, ConsumerIgnoresMetadata) {
    auto ch = ctx->create_channel();
    std::vector<Message> data_outputs;
    std::vector<Actor> actors;

    auto producer = [this, ch]() -> Actor {
        ShutdownAtExit c{ch};
        co_await ctx->executor()->schedule();

        co_await ch->send_metadata(
            Message{0, std::make_unique<int>(10), rapidsmpf::ContentDescription{}}
        );
        co_await ch->send_metadata(
            Message{0, std::make_unique<int>(20), rapidsmpf::ContentDescription{}}
        );
        co_await ch->send(
            Message{1, std::make_unique<int>(30), rapidsmpf::ContentDescription{}}
        );
        co_await ch->drain(ctx->executor());
    };

    auto consumer = [this, ch, &data_outputs]() -> Actor {
        ShutdownAtExit c{ch};
        co_await ctx->executor()->schedule();
        while (true) {
            auto msg = co_await ch->receive();
            if (msg.empty()) {
                break;
            }
            data_outputs.push_back(std::move(msg));
        }
    };

    actors.emplace_back(producer());
    actors.emplace_back(consumer());
    run_actor_graph(std::move(actors));

    EXPECT_EQ(data_outputs.size(), 1U);
    EXPECT_EQ(data_outputs[0].get<int>(), 30);
}

TEST_F(StreamingChannel, ProducerThrowsWithMetadata) {
    auto ch = ctx->create_channel();
    std::vector<Actor> actors;

    auto producer = [this, ch]() -> Actor {
        ShutdownAtExit c{ch};
        co_await ctx->executor()->schedule();
        co_await ch->send_metadata(
            Message{0, std::make_unique<int>(31), rapidsmpf::ContentDescription{}}
        );
        throw std::runtime_error("producer failed");
    };

    auto consumer = [this, ch]() -> Actor {
        ShutdownAtExit c{ch};
        co_await ctx->executor()->schedule();
        while (true) {
            auto msg = co_await ch->receive_metadata();
            if (msg.empty()) {
                break;
            }
        }
    };

    actors.emplace_back(producer());
    actors.emplace_back(consumer());
    EXPECT_THROW(run_actor_graph(std::move(actors)), std::runtime_error);
}

TEST_F(StreamingChannel, ConsumerThrowsWithMetadata) {
    auto ch = ctx->create_channel();
    std::vector<Actor> actors;

    auto producer = [this, ch]() -> Actor {
        ShutdownAtExit c{ch};
        co_await ctx->executor()->schedule();
        co_await ch->send_metadata(
            Message{0, std::make_unique<int>(10), rapidsmpf::ContentDescription{}}
        );
        co_await ch->drain(ctx->executor());
    };

    auto consumer = [this, ch]() -> Actor {
        ShutdownAtExit c{ch};
        co_await ctx->executor()->schedule();
        throw std::runtime_error("consumer failed");
    };

    actors.emplace_back(producer());
    actors.emplace_back(consumer());
    EXPECT_THROW(run_actor_graph(std::move(actors)), std::runtime_error);
}

TEST_F(StreamingChannel, ProducerAndConsumerThrow) {
    auto ch = ctx->create_channel();
    std::vector<Actor> actors;

    auto producer = [this, ch]() -> Actor {
        ShutdownAtExit c{ch};
        co_await ctx->executor()->schedule();
        throw std::runtime_error("producer failed");
    };

    auto consumer = [this, ch]() -> Actor {
        ShutdownAtExit c{ch};
        co_await ctx->executor()->schedule();
        throw std::runtime_error("consumer failed");
    };

    actors.emplace_back(producer());
    actors.emplace_back(consumer());
    EXPECT_THROW(run_actor_graph(std::move(actors)), std::runtime_error);
}
