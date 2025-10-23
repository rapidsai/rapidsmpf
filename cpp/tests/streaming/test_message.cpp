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
    Message m{std::make_unique<int>(42)};
    EXPECT_FALSE(m.empty());
    EXPECT_TRUE(m.holds<int>());
    EXPECT_FALSE(m.holds<std::string>());
    EXPECT_EQ(m.get<int>(), 42);
    EXPECT_THROW(m.get<std::string>(), std::invalid_argument);
}

TEST_F(StreamingMessage, ReleaseEmpties) {
    Message m{std::make_unique<std::string>("abc")};
    auto s = m.release<std::string>();
    EXPECT_EQ(s, "abc");
    EXPECT_TRUE(m.empty());
}

TEST_F(StreamingMessage, ResetEmpties) {
    Message m{std::make_unique<std::string>("abc")};
    EXPECT_EQ(m.get<std::string>(), "abc");
    m.reset();
    EXPECT_TRUE(m.empty());
}

TEST_F(StreamingMessage, BufferSizeWithoutCallbacks) {
    Message m{br->allocate(stream, br->reserve_or_fail(10, MemoryType::HOST))};
    EXPECT_THROW(std::ignore = m.buffer_size(MemoryType::HOST), std::invalid_argument);
    EXPECT_THROW(std::ignore = m.buffer_size(MemoryType::DEVICE), std::invalid_argument);
}

TEST_F(StreamingMessage, BufferSizeWithCallbacks) {
    Message::Callbacks cbs{
        .buffer_size = [](Message const& msg, MemoryType mem_type) -> size_t {
            EXPECT_TRUE(msg.holds<Buffer>());
            if (mem_type == msg.get<Buffer>().mem_type()) {
                return msg.get<Buffer>().size;
            }
            return 0;
        }
    };

    // Host memory
    {
        Message m{br->allocate(stream, br->reserve_or_fail(10, MemoryType::HOST)), cbs};
        EXPECT_EQ(m.buffer_size(MemoryType::HOST), 10);
        EXPECT_EQ(m.buffer_size(MemoryType::DEVICE), 0);
    }

    // Device memory
    {
        Message m{br->allocate(stream, br->reserve_or_fail(10, MemoryType::DEVICE)), cbs};
        EXPECT_EQ(m.buffer_size(MemoryType::HOST), 0);
        EXPECT_EQ(m.buffer_size(MemoryType::DEVICE), 10);
    }
}
