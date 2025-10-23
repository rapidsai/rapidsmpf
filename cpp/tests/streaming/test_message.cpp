/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rapidsmpf/streaming/core/message.hpp>


using namespace rapidsmpf;
using namespace rapidsmpf::streaming;

TEST(StreamingMessage, ConstructAndGetInt) {
    Message m{std::make_unique<int>(42)};
    EXPECT_FALSE(m.empty());
    EXPECT_TRUE(m.holds<int>());
    EXPECT_FALSE(m.holds<std::string>());
    EXPECT_EQ(m.get<int>(), 42);
    EXPECT_THROW(m.get<std::string>(), std::invalid_argument);
}

TEST(StreamingMessage, ReleaseEmpties) {
    Message m{std::make_unique<std::string>("abc")};
    auto s = m.release<std::string>();
    EXPECT_EQ(s, "abc");
    EXPECT_TRUE(m.empty());
}

TEST(StreamingMessage, ResetEmpties) {
    Message m{std::make_unique<std::string>("abc")};
    EXPECT_EQ(m.get<std::string>(), "abc");
    m.reset();
    EXPECT_TRUE(m.empty());
}
