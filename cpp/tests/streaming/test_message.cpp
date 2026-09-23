/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cuda/stream>

#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <rapidsmpf/streaming/chunks/partition.hpp>
#include <rapidsmpf/streaming/core/message.hpp>

using namespace rapidsmpf;
using namespace rapidsmpf::streaming;

class StreamingMessage : public ::testing::Test {
  protected:
    void SetUp() override {
        br = BufferResource::create(rmm::mr::get_current_device_resource_ref());
        stream = cuda::stream_ref{cudaStreamLegacy};
    }

    std::shared_ptr<BufferResource> br;
    cuda::stream_ref stream{cudaStreamLegacy};
};

TEST_F(StreamingMessage, ConstructAndGetInt) {
    auto payload = std::make_unique<int>(42);
    Message m{0, std::move(payload), ContentDescription{}};
    EXPECT_FALSE(m.empty());
    EXPECT_TRUE(m.holds<int>());
    EXPECT_FALSE(m.holds<std::string>());
    EXPECT_EQ(m.get<int>(), 42);
    EXPECT_THROW(std::ignore = m.get<std::string>(), std::invalid_argument);
}

TEST_F(StreamingMessage, ReleaseEmpties) {
    auto payload = std::make_unique<std::string>("abc");
    Message m{0, std::move(payload), ContentDescription{}};
    auto s = m.release<std::string>();
    EXPECT_EQ(s, "abc");
    EXPECT_TRUE(m.empty());
}

TEST_F(StreamingMessage, ResetEmpties) {
    auto payload = std::make_unique<std::string>("abc");
    Message m{0, std::move(payload), ContentDescription{}};
    EXPECT_EQ(m.get<std::string>(), "abc");
    m.reset();
    EXPECT_TRUE(m.empty());
}

TEST_F(StreamingMessage, ContentSize) {
    // Test `content_size`, ignore the payload (we use an int as a dummy).
    {
        ContentDescription cd{
            {{MemoryType::HOST, 10}}, ContentDescription::Spillable::YES
        };
        Message m{0, std::make_unique<int>(42), cd};
        EXPECT_TRUE(m.content_description().spillable());
        EXPECT_EQ(m.content_description().content_size(MemoryType::HOST), 10);
        EXPECT_EQ(m.content_description().content_size(MemoryType::DEVICE), 0);
    }
    {
        ContentDescription cd{
            {{MemoryType::HOST, 10}, {MemoryType::DEVICE, 20}},
            ContentDescription::Spillable::NO
        };
        Message m{0, std::make_unique<int>(42), cd};
        EXPECT_FALSE(m.content_description().spillable());
        EXPECT_EQ(m.content_description().content_size(MemoryType::HOST), 10);
        EXPECT_EQ(m.content_description().content_size(MemoryType::DEVICE), 20);
    }
}

TEST_F(StreamingMessage, CopyWithoutCallbacks) {
    Message m{
        0,
        br->make_buffer(stream, br->reserve_or_fail(10, MemoryType::HOST)),
        ContentDescription{}
    };
    {
        auto res = br->reserve_or_fail(m.copy_cost(), MemoryType::HOST);
        EXPECT_THROW(std::ignore = m.copy(res), std::invalid_argument);
    }
    {
        auto res = br->reserve_or_fail(m.copy_cost(), MemoryType::DEVICE);
        EXPECT_THROW(std::ignore = m.copy(res), std::invalid_argument);
    }
}

TEST_F(StreamingMessage, CopyWithCallbacks) {
    Message::CopyCallback copy_cb = [](Message const& msg,
                                       MemoryReservation& reservation) -> Message {
        EXPECT_TRUE(msg.holds<Buffer>());
        auto const& src = msg.get<Buffer>();
        auto dst = reservation.br()->make_buffer(src.size, src.stream(), reservation);
        buffer_copy(reservation.br()->statistics(), *dst, src, src.size);
        ContentDescription cd{
            {{dst->mem_type(), dst->size}}, ContentDescription::Spillable::YES
        };
        return Message{msg.sequence_number(), std::move(dst), cd, msg.copy_cb()};
    };
    {
        ContentDescription cd{
            {{MemoryType::HOST, 10}}, ContentDescription::Spillable::YES
        };
        Message m1{
            42,
            br->make_buffer(stream, br->reserve_or_fail(10, MemoryType::HOST)),
            cd,
            copy_cb
        };
        EXPECT_EQ(m1.copy_cost(), 10);
        auto res = br->reserve_or_fail(m1.copy_cost(), MemoryType::HOST);
        auto m2 = m1.copy(res);
        EXPECT_EQ(m1.get<Buffer>().mem_type(), m2.get<Buffer>().mem_type());
        EXPECT_EQ(m1.get<Buffer>().size, m2.get<Buffer>().size);
        EXPECT_EQ(m1.sequence_number(), m2.sequence_number());
    }
    {
        ContentDescription cd{
            {{MemoryType::DEVICE, 10}}, ContentDescription::Spillable::YES
        };
        Message m1{
            42,
            br->make_buffer(stream, br->reserve_or_fail(10, MemoryType::DEVICE)),
            cd,
            copy_cb
        };
        EXPECT_EQ(m1.copy_cost(), 10);
        auto res = br->reserve_or_fail(m1.copy_cost(), MemoryType::DEVICE);
        auto m2 = m1.copy(res);
        EXPECT_EQ(m1.get<Buffer>().mem_type(), m2.get<Buffer>().mem_type());
        EXPECT_EQ(m1.sequence_number(), m2.sequence_number());
    }
}

namespace {

/// @brief Callbacks that count their invocations and rebuild an `int` message.
Message::Callbacks counting_callbacks(int& copies, int& moves) {
    return {
        .copy = [&copies](Message const& msg, MemoryReservation&) -> Message {
            ++copies;
            return Message{
                msg.sequence_number(),
                std::make_unique<int>(msg.get<int>()),
                msg.content_description(),
                msg.callbacks()
            };
        },
        .move = [&moves](Message&& msg, MemoryReservation&) -> Message {
            ++moves;
            auto callbacks = msg.callbacks();
            auto cd = msg.content_description();
            auto payload = std::make_unique<int>(msg.release<int>());
            return Message{msg.sequence_number(), std::move(payload), cd, callbacks};
        }
    };
}

}  // namespace

TEST_F(StreamingMessage, CopyAndMoveUseTheirOwnCallbacks) {
    int copies = 0;
    int moves = 0;
    Message m{
        7,
        std::make_unique<int>(42),
        ContentDescription{},
        counting_callbacks(copies, moves)
    };
    auto res = br->reserve_or_fail(0, MemoryType::HOST);

    auto copied = m.copy(res);
    EXPECT_EQ(copies, 1);
    EXPECT_EQ(moves, 0);
    EXPECT_EQ(copied.get<int>(), 42);

    auto moved = m.move(res);
    EXPECT_EQ(copies, 1);
    EXPECT_EQ(moves, 1);
    EXPECT_TRUE(m.empty());
    EXPECT_EQ(moved.get<int>(), 42);
}

TEST_F(StreamingMessage, MoveFallsBackToCopy) {
    int copies = 0;
    int moves = 0;
    Message m{
        7,
        std::make_unique<int>(42),
        ContentDescription{},
        counting_callbacks(copies, moves).copy
    };
    auto res = br->reserve_or_fail(0, MemoryType::HOST);

    auto moved = m.move(res);
    EXPECT_EQ(copies, 1);
    EXPECT_EQ(moves, 0);
    EXPECT_TRUE(m.empty());  // Reset even though the copy left it intact.
    EXPECT_EQ(moved.get<int>(), 42);
}

TEST_F(StreamingMessage, AThrowingMoveStillResets) {
    // The callback throws before touching the payload, and the message is reset anyway.
    Message m{
        0,
        std::make_unique<int>(42),
        ContentDescription{},
        Message::Callbacks{
            .copy = nullptr, .move = [](Message&&, MemoryReservation&) -> Message {
                throw std::runtime_error("move failed");
            }
        }
    };
    auto res = br->reserve_or_fail(0, MemoryType::HOST);
    EXPECT_THROW(std::ignore = m.move(res), std::runtime_error);
    EXPECT_TRUE(m.empty());
}

TEST_F(StreamingMessage, MoveWithoutCallbacks) {
    Message m{0, std::make_unique<int>(42), ContentDescription{}};
    auto res = br->reserve_or_fail(0, MemoryType::HOST);
    EXPECT_THROW(std::ignore = m.move(res), std::invalid_argument);
    EXPECT_FALSE(m.empty());  // Untouched when neither callback exists.
}
