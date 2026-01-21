/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rapidsmpf/utils.hpp>

using namespace rapidsmpf;

TEST(UtilsTest, ParseStringTest) {
    // Integers
    EXPECT_EQ(parse_string<int>("42"), 42);
    EXPECT_EQ(parse_string<int>("-7"), -7);
    EXPECT_THROW(parse_string<int>("abc"), std::invalid_argument);

    // Doubles
    EXPECT_DOUBLE_EQ(parse_string<double>("3.14"), 3.14);
    EXPECT_DOUBLE_EQ(parse_string<double>("-2.71"), -2.71);
    EXPECT_THROW(parse_string<double>("not_a_double"), std::invalid_argument);

    // Booleans from integers
    EXPECT_TRUE(parse_string<bool>("1"));
    EXPECT_FALSE(parse_string<bool>("0"));
    EXPECT_TRUE(parse_string<bool>("42"));  // non-zero is true

    // Booleans from text
    EXPECT_TRUE(parse_string<bool>("true"));
    EXPECT_TRUE(parse_string<bool>("yes"));
    EXPECT_TRUE(parse_string<bool>("on"));
    EXPECT_FALSE(parse_string<bool>("false"));
    EXPECT_FALSE(parse_string<bool>("no"));
    EXPECT_FALSE(parse_string<bool>("off"));

    // Case and whitespace handling
    EXPECT_TRUE(parse_string<bool>(" YES "));
    EXPECT_FALSE(parse_string<bool>("\toFf\n"));

    // Invalid boolean
    EXPECT_THROW(parse_string<bool>("not_a_bool"), std::invalid_argument);
}
