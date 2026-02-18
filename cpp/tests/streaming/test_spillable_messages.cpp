/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rapidsmpf/streaming/core/spillable_messages.hpp>

#include "base_streaming_fixture.hpp"

using namespace rapidsmpf;
using namespace rapidsmpf::streaming;

class StreamingSpillableMessages : public BaseStreamingFixture {
  public:
    void SetUp() override {
        SetUpWithThreads(8);
    }

    void TearDown() override {
        BaseStreamingFixture::TearDown();
    }
};

/**
 * @brief Create a simple integer message for testing.
 *
 * @param sequence_number Sequence number assigned to the message.
 * @param payload Integer payload value.
 * @param mem_type Memory type associated with the payload.
 * @param spillable Indicates whether the message content is spillable.
 * @return A new `Message` instance containing the integer payload.
 */
Message create_int_msg(
    std::uint64_t sequence_number,
    int payload,
    MemoryType mem_type,
    ContentDescription::Spillable spillable
) {
    auto cd = ContentDescription{{{mem_type, sizeof(int)}}, spillable};
    return {
        sequence_number,
        std::make_unique<int>(payload),
        cd,
        [](Message const& msg, MemoryReservation& reservation) -> Message {
            auto const& self = msg.get<int>();
            // We never copy to device memory, we just pretend.
            auto cd = ContentDescription{
                {{reservation.mem_type(), sizeof(int)}},
                msg.content_description().spillable() ? ContentDescription::Spillable::YES
                                                      : ContentDescription::Spillable::NO
            };
            return Message{
                msg.sequence_number(), std::make_unique<int>(self), cd, msg.copy_cb()
            };
        }
    };
}

TEST_F(StreamingSpillableMessages, ExtractError) {
    SpillableMessages msgs;
    EXPECT_THROW(std::ignore = msgs.extract(0), std::out_of_range);
}

TEST_F(StreamingSpillableMessages, InsertSpillExtract) {
    SpillableMessages msgs;
    EXPECT_TRUE(msgs.get_content_descriptions().empty());

    auto mid1 = msgs.insert(
        create_int_msg(1, 2, MemoryType::DEVICE, ContentDescription::Spillable::YES)
    );
    EXPECT_EQ(msgs.get_content_descriptions().size(), 1);
    EXPECT_EQ(
        msgs.get_content_descriptions().at(mid1).content_size(MemoryType::DEVICE),
        sizeof(int)
    );
    EXPECT_EQ(msgs.spill(mid1, br.get()), sizeof(int));
    EXPECT_EQ(msgs.spill(mid1, br.get()), 0);
    EXPECT_EQ(msgs.spill(123, br.get()), 0);

    auto mid2 = msgs.insert(
        create_int_msg(2, 3, MemoryType::DEVICE, ContentDescription::Spillable::NO)
    );
    EXPECT_EQ(msgs.get_content_descriptions().size(), 2);
    EXPECT_EQ(
        msgs.get_content_descriptions().at(mid2).content_size(MemoryType::DEVICE),
        sizeof(int)
    );
    EXPECT_EQ(msgs.spill(mid2, br.get()), 0);

    auto msg = msgs.extract(mid1);
    EXPECT_EQ(msg.sequence_number(), 1);
    EXPECT_EQ(msg.get<int>(), 2);
    EXPECT_THROW(std::ignore = msgs.extract(mid1), std::out_of_range);
    EXPECT_EQ(msgs.get_content_descriptions().size(), 1);

    msg = msgs.extract(mid2);
    EXPECT_EQ(msg.sequence_number(), 2);
    EXPECT_EQ(msg.get<int>(), 3);
    EXPECT_THROW(std::ignore = msgs.extract(mid2), std::out_of_range);
    EXPECT_TRUE(msgs.get_content_descriptions().empty());
}

TEST_F(StreamingSpillableMessages, MultiThreadedRandomInsertSpillExtract) {
    using MID = SpillableMessages::MessageId;

    constexpr int num_producers = 5;
    constexpr int msgs_per_producer = 1000;
    constexpr int num_spillers = 10;
    constexpr int num_consumers = 10;
    constexpr int total_msgs = num_producers * msgs_per_producer;

    SpillableMessages msgs;
    std::atomic<int> extracted_sum{0};
    std::vector<std::thread> threads;

    // Producers: insert messages with payload=1 and spillable=YES.
    for (int i = 0; i < num_producers; ++i) {
        threads.emplace_back([&msgs, i] {
            for (int j = 0; j < msgs_per_producer; ++j) {
                std::ignore = msgs.insert(create_int_msg(
                    msgs_per_producer * i + j,
                    1,
                    MemoryType::DEVICE,
                    ContentDescription::Spillable::YES
                ));
            }
        });
    }

    // Spillers: randomly try to spill IDs in [0, total_msgs]; OK if non-existent.
    for (int i = 0; i < num_spillers; ++i) {
        threads.emplace_back([this, &msgs, i] {
            std::mt19937 rng{static_cast<std::mt19937::result_type>(0xD00 + i)};
            std::uniform_int_distribution<MID> dist(0, total_msgs * num_spillers);
            for (int j = 0; j < total_msgs; ++j) {
                std::ignore = msgs.spill(dist(rng), br.get());
            }
        });
    }

    // Consumers: randomly try to extract IDs in [0, total_msgs]; ignore misses.
    for (int i = 0; i < num_consumers; ++i) {
        threads.emplace_back([&msgs, &extracted_sum, i] {
            std::mt19937 rng{static_cast<std::mt19937::result_type>(0xC00 + i)};
            std::uniform_int_distribution<MID> dist(0, total_msgs);
            for (int j = 0; j < total_msgs; ++j) {
                try {
                    auto msg = msgs.extract(dist(rng));
                    extracted_sum += msg.get<int>();  // payload is 1
                } catch (std::out_of_range const&) {
                    // Already extracted or never existed — ignore
                }
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // Final sweep.
    for (MID mid = 0; mid <= static_cast<MID>(total_msgs); ++mid) {
        try {
            auto msg = msgs.extract(mid);
            extracted_sum += msg.get<int>();
        } catch (std::out_of_range const&) {
            // Already extracted or never existed — ignore
        }
    }

    // All inserted messages had payload=1
    EXPECT_EQ(extracted_sum.load(), total_msgs);
    EXPECT_TRUE(msgs.get_content_descriptions().empty());
}

TEST_F(StreamingSpillableMessages, SpillInFlightMessages) {
    constexpr int num_producer_producer_pairs = 16;
    constexpr int msgs_per_producer = 100;
    std::vector<Actor> actors;
    std::atomic<std::size_t> spilled_expect{0};
    std::atomic<std::size_t> spilled_got{0};

    // Create many producer-consumer pairs and spill in-flight messages.
    for (int i = 0; i < num_producer_producer_pairs; ++i) {
        auto ch = ctx->create_channel();
        actors.emplace_back(
            [](int i,
               std::shared_ptr<Context> ctx,
               std::shared_ptr<Channel> ch_out) -> Actor {
                ShutdownAtExit c{ch_out};
                co_await ctx->executor()->schedule();
                for (int j = 0; j < msgs_per_producer; ++j) {
                    auto seq = i * num_producer_producer_pairs + j;
                    co_await ch_out->send(create_int_msg(
                        seq, seq, MemoryType::DEVICE, ContentDescription::Spillable::YES
                    ));
                }
                co_await ch_out->drain(ctx->executor());
            }(i, ctx, ch)
        );
        actors.emplace_back(
            [](int i,
               std::atomic<std::size_t>& spilled_expect,
               std::atomic<std::size_t>& spilled_got,
               std::shared_ptr<Context> ctx,
               std::shared_ptr<Channel> ch_in) -> Actor {
                ShutdownAtExit c{ch_in};
                co_await ctx->executor()->schedule();
                for (int j = 0; j < msgs_per_producer; ++j) {
                    spilled_expect.fetch_add(
                        ctx->br()->spill_manager().spill(1024), std::memory_order_relaxed
                    );

                    auto msg = co_await ch_in->receive();
                    if (msg.empty()) {
                        break;
                    }

                    spilled_got.fetch_add(
                        msg.content_description().content_size(MemoryType::HOST),
                        std::memory_order_relaxed
                    );
                    auto seq = i * num_producer_producer_pairs + j;
                    EXPECT_EQ(msg.sequence_number(), seq);
                    EXPECT_EQ(msg.release<int>(), seq);
                }
            }(i, spilled_expect, spilled_got, ctx, ch)
        );
    }

    // Randomize the actor order.
    std::shuffle(actors.begin(), actors.end(), std::mt19937{std::random_device{}()});
    run_actor_graph(std::move(actors));

    EXPECT_EQ(
        spilled_expect.load(std::memory_order_relaxed),
        spilled_got.load(std::memory_order_relaxed)
    );
}

TEST_F(StreamingSpillableMessages, CopyInvalidIdThrows) {
    SpillableMessages msgs;

    // Try to copy a non-existent ID should throw.
    auto reservation = br->reserve_or_fail(10, MemoryType::DEVICE);
    EXPECT_THROW(std::ignore = msgs.copy(1234, reservation), std::out_of_range);
}

TEST_F(StreamingSpillableMessages, Copy) {
    SpillableMessages msgs;

    // Original message is HOST-backed and spillable.
    auto mid = msgs.insert(
        create_int_msg(1, 42, MemoryType::HOST, ContentDescription::Spillable::YES)
    );

    // Sanity check of the initial content description.
    auto cds_before = msgs.get_content_descriptions();
    ASSERT_EQ(cds_before.size(), 1);
    EXPECT_EQ(cds_before.at(mid).content_size(MemoryType::HOST), sizeof(int));
    EXPECT_EQ(cds_before.at(mid).content_size(MemoryType::DEVICE), 0);

    // Let's copy.
    auto reservation = br->reserve_or_fail(10, MemoryType::DEVICE);
    auto copy = msgs.copy(mid, reservation);

    // Copied message has same sequence number and payload.
    EXPECT_EQ(copy.sequence_number(), 1);
    EXPECT_EQ(copy.get<int>(), 42);

    // Copied message should now report DEVICE-backed content, not HOST.
    EXPECT_EQ(copy.content_description().content_size(MemoryType::DEVICE), sizeof(int));
    EXPECT_EQ(copy.content_description().content_size(MemoryType::HOST), 0);

    // Original message is still present and unchanged in the container.
    auto cds_after = msgs.get_content_descriptions();
    ASSERT_EQ(cds_after.size(), 1);
    EXPECT_EQ(cds_after.at(mid).content_size(MemoryType::HOST), sizeof(int));
    EXPECT_EQ(cds_after.at(mid).content_size(MemoryType::DEVICE), 0);

    // We can still extract the original message and it has the same payload.
    auto original = msgs.extract(mid);
    EXPECT_EQ(original.sequence_number(), 1);
    EXPECT_EQ(original.get<int>(), 42);
}
