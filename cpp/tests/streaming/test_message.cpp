/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

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
    Message m{0, std::make_unique<int>(42)};
    EXPECT_FALSE(m.empty());
    EXPECT_TRUE(m.holds<int>());
    EXPECT_FALSE(m.holds<std::string>());
    EXPECT_EQ(m.get<int>(), 42);
    EXPECT_THROW(m.get<std::string>(), std::invalid_argument);
}

TEST_F(StreamingMessage, ReleaseEmpties) {
    Message m{0, std::make_unique<std::string>("abc")};
    auto s = m.release<std::string>();
    EXPECT_EQ(s, "abc");
    EXPECT_TRUE(m.empty());
}

TEST_F(StreamingMessage, ResetEmpties) {
    Message m{0, std::make_unique<std::string>("abc")};
    EXPECT_EQ(m.get<std::string>(), "abc");
    m.reset();
    EXPECT_TRUE(m.empty());
}

TEST_F(StreamingMessage, BufferSizeWithoutCallbacks) {
    Message m{0, br->allocate(stream, br->reserve_or_fail(10, MemoryType::HOST))};
    EXPECT_THROW(
        std::ignore = m.primary_data_size(MemoryType::HOST), std::invalid_argument
    );
    EXPECT_THROW(
        std::ignore = m.primary_data_size(MemoryType::DEVICE), std::invalid_argument
    );
}

TEST_F(StreamingMessage, BufferSizeWithCallbacks) {
    Message::Callbacks cbs{
        .primary_data_size = [](Message const& msg,
                                MemoryType mem_type) -> std::pair<size_t, bool> {
            EXPECT_TRUE(msg.holds<Buffer>());
            if (mem_type == msg.get<Buffer>().mem_type()) {
                return {msg.get<Buffer>().size, true};
            }
            return {0, false};
        }
    };
    {
        Message m{
            0, br->allocate(stream, br->reserve_or_fail(10, MemoryType::HOST)), cbs
        };
        EXPECT_EQ(
            m.primary_data_size(MemoryType::HOST), std::make_pair(size_t{10}, true)
        );
        EXPECT_EQ(
            m.primary_data_size(MemoryType::DEVICE), std::make_pair(size_t{0}, false)
        );
    }
    {
        Message m{
            0, br->allocate(stream, br->reserve_or_fail(10, MemoryType::DEVICE)), cbs
        };
        EXPECT_EQ(
            m.primary_data_size(MemoryType::HOST), std::make_pair(size_t{0}, false)
        );
        EXPECT_EQ(
            m.primary_data_size(MemoryType::DEVICE), std::make_pair(size_t{10}, true)
        );
    }
}

TEST_F(StreamingMessage, CopyWithoutCallbacks) {
    Message m{0, br->allocate(stream, br->reserve_or_fail(10, MemoryType::HOST))};
    {
        auto res = br->reserve_or_fail(10, MemoryType::HOST);
        EXPECT_THROW(std::ignore = m.copy(br.get(), res), std::invalid_argument);
    }
    {
        auto res = br->reserve_or_fail(10, MemoryType::DEVICE);
        EXPECT_THROW(std::ignore = m.copy(br.get(), res), std::invalid_argument);
    }
}

TEST_F(StreamingMessage, CopyWithCallbacks) {
    Message::Callbacks cbs{
        .copy = [](Message const& msg,
                   BufferResource* br,
                   MemoryReservation& reservation) -> Message {
            EXPECT_TRUE(msg.holds<Buffer>());
            auto const& src = msg.get<Buffer>();
            auto dst = br->allocate(src.size, src.stream(), reservation);
            buffer_copy(*dst, src, src.size);
            return Message(msg.sequence_number(), std::move(dst), msg.callbacks());
        }
    };
    {
        Message m1{
            42, br->allocate(stream, br->reserve_or_fail(10, MemoryType::HOST)), cbs
        };
        auto res = br->reserve_or_fail(10, MemoryType::HOST);
        auto m2 = m1.copy(br.get(), res);
        EXPECT_EQ(m1.get<Buffer>().mem_type(), m2.get<Buffer>().mem_type());
        EXPECT_EQ(m1.get<Buffer>().size, m2.get<Buffer>().size);
        EXPECT_EQ(m1.sequence_number(), m2.sequence_number());
    }
    {
        Message m1{
            42, br->allocate(stream, br->reserve_or_fail(10, MemoryType::DEVICE)), cbs
        };
        auto res = br->reserve_or_fail(10, MemoryType::DEVICE);
        auto m2 = m1.copy(br.get(), res);
        EXPECT_EQ(m1.get<Buffer>().mem_type(), m2.get<Buffer>().mem_type());
        EXPECT_EQ(m1.sequence_number(), m2.sequence_number());
    }
}
