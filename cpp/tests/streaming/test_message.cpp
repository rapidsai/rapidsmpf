/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rapidsmpf/streaming/core/channel.hpp>


using namespace rapidsmpf;
using namespace rapidsmpf::streaming;

// Basic construction and type/int access
TEST(StreamingMessage, ConstructAndGetInt) {
    Message m{std::make_unique<int>(42)};
    EXPECT_FALSE(m.empty());
    EXPECT_TRUE(m.holds<int>());
    EXPECT_EQ(m.get<int>(), 42);
}

// Releasing moves out and empties when never accessed (sole-ownership not required)
TEST(StreamingMessage, ReleaseEmpties) {
    Message m{std::make_unique<std::string>("abc")};
    auto s = m.release<std::string>();
    EXPECT_EQ(s, "abc");
    EXPECT_TRUE(m.empty());
    // holds<T>() on empty should throw per API
    EXPECT_THROW(std::ignore = m.holds<std::string>(), std::invalid_argument);
}

// Shallow copy shares payload; release is only allowed if sole owner after access
TEST(StreamingMessage, ShallowCopySharesAndReleaseRules) {
    Message m{std::make_unique<int>(7)};
    ASSERT_FALSE(m.empty());

    // Access marks as "accessed"
    EXPECT_EQ(m.get<int>(), 7);

    // Shallow copy always allowed; both share the same payload
    auto m2 = m.shallow_copy();
    EXPECT_TRUE(m.holds<int>());
    EXPECT_TRUE(m2.holds<int>());
    EXPECT_EQ(&m.get<int>(), &m2.get<int>());

    // Since payload was accessed and there are multiple owners, release must fail
    EXPECT_THROW(std::ignore = m.release<int>(), std::invalid_argument);
    EXPECT_THROW(std::ignore = m2.release<int>(), std::invalid_argument);

    // Drop one owner; now sole owner and previously accessed -> release succeeds
    m2.reset();
    EXPECT_NO_THROW({
        int v = m.release<int>();
        EXPECT_EQ(v, 7);
    });
}

// Type mismatch behavior and reset semantics
TEST(StreamingMessage, TypeMismatchAndReset) {
    Message m{std::make_unique<int>(5)};
    EXPECT_FALSE(m.empty());
    EXPECT_FALSE(m.holds<std::string>());  // wrong type => false (message not empty)

    // get with wrong type throws
    EXPECT_THROW(std::ignore = m.get<std::string>(), std::invalid_argument);

    // reset empties the message
    m.reset();
    EXPECT_TRUE(m.empty());
    EXPECT_THROW(std::ignore = m.holds<int>(), std::invalid_argument);
    EXPECT_THROW(std::ignore = m.get<int>(), std::invalid_argument);
}

// Constructing with null unique_ptr is invalid
TEST(StreamingMessage, NullUniquePtrThrows) {
    std::unique_ptr<int> p{};
    EXPECT_THROW(std::ignore = Message{std::move(p)}, std::invalid_argument);
}

// Move semantics leave source empty; destination retains payload
TEST(StreamingMessage, MoveSemanticsLeaveSourceEmpty) {
    Message a{std::make_unique<int>(99)};
    Message b = std::move(a);
    EXPECT_TRUE(a.empty());
    EXPECT_FALSE(b.empty());
    EXPECT_EQ(b.get<int>(), 99);

    Message c{std::make_unique<int>(1)};
    b = std::move(c);
    EXPECT_TRUE(c.empty());
    EXPECT_EQ(b.get<int>(), 1);
}
