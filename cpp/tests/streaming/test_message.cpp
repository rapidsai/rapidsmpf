/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rapidsmpf/buffer/resource.hpp>
#include <rapidsmpf/streaming/chunks/partition.hpp>
#include <rapidsmpf/streaming/core/message.hpp>

using namespace rapidsmpf;
using namespace rapidsmpf::streaming;

class StreamingMessage : public ::testing::Test {
  protected:
    void SetUp() override {
        br = std::make_unique<BufferResource>(cudf::get_current_device_resource_ref());
        stream = cudf::get_default_stream();
    }

    std::unique_ptr<BufferResource> br;
    rmm::cuda_stream_view stream;
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
        br->allocate(stream, br->reserve_or_fail(10, MemoryType::HOST)),
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
        auto dst = reservation.br()->allocate(src.size, src.stream(), reservation);
        buffer_copy(*dst, src, src.size);
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
            br->allocate(stream, br->reserve_or_fail(10, MemoryType::HOST)),
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
            br->allocate(stream, br->reserve_or_fail(10, MemoryType::DEVICE)),
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
// test context description properties
constexpr ContentDescription cd{
    {{MemoryType::HOST, 10}, {MemoryType::DEVICE, 20}}, ContentDescription::Spillable::NO
};
static_assert(cd.highest_memory_type_set() == MemoryType::DEVICE);
static_assert(cd.lowest_memory_type_set() == MemoryType::HOST);
static_assert(std::ranges::equal(
    LowerMemoryTypesInclusive(cd.lowest_memory_type_set()),
    std::span<const MemoryType>(MEMORY_TYPES).subspan(1)  // [HOST]
));

constexpr ContentDescription cd2{
    {{MemoryType::DEVICE, 10}}, ContentDescription::Spillable::NO
};
static_assert(cd2.highest_memory_type_set() == MemoryType::DEVICE);
static_assert(cd2.lowest_memory_type_set() == MemoryType::DEVICE);
static_assert(std::ranges::equal(
    LowerMemoryTypesInclusive(cd2.lowest_memory_type_set()),
    std::span<const MemoryType>(MEMORY_TYPES)
));

constexpr ContentDescription cd3{
    {{MemoryType::HOST, 10}}, ContentDescription::Spillable::NO
};
static_assert(cd3.highest_memory_type_set() == MemoryType::HOST);
static_assert(cd3.lowest_memory_type_set() == MemoryType::HOST);
static_assert(std::ranges::equal(
    LowerMemoryTypesInclusive(cd3.lowest_memory_type_set()),
    std::span<const MemoryType>(MEMORY_TYPES).subspan(1)  // [HOST]
));

constexpr ContentDescription cd4{};
static_assert(cd4.highest_memory_type_set() == MemoryType::HOST);
static_assert(cd4.lowest_memory_type_set() == MemoryType::HOST);
static_assert(std::ranges::equal(
    LowerMemoryTypesInclusive(cd4.lowest_memory_type_set()),
    std::span<const MemoryType>(MEMORY_TYPES).subspan(1)  // [HOST]
));
}  // namespace
