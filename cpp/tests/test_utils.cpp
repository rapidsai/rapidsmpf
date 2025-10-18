/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rapidsmpf/utils.hpp>

#if RAPIDSMPF_HAVE_NUMA
#include <numa.h>
#endif

#include <iostream>

using namespace rapidsmpf;

TEST(ParseStringTest, ParsesIntegers) {
    EXPECT_EQ(parse_string<int>("42"), 42);
    EXPECT_EQ(parse_string<int>("-7"), -7);
    EXPECT_THROW(parse_string<int>("abc"), std::invalid_argument);
}

TEST(ParseStringTest, ParsesDoubles) {
    EXPECT_DOUBLE_EQ(parse_string<double>("3.14"), 3.14);
    EXPECT_DOUBLE_EQ(parse_string<double>("-2.71"), -2.71);
    EXPECT_THROW(parse_string<double>("not_a_double"), std::invalid_argument);
}

TEST(ParseStringTest, ParsesBooleansFromIntegers) {
    EXPECT_TRUE(parse_string<bool>("1"));
    EXPECT_FALSE(parse_string<bool>("0"));
    EXPECT_TRUE(parse_string<bool>("42"));  // non-zero is true
}

TEST(ParseStringTest, ParsesBooleansFromText) {
    EXPECT_TRUE(parse_string<bool>("true"));
    EXPECT_TRUE(parse_string<bool>("yes"));
    EXPECT_TRUE(parse_string<bool>("on"));

    EXPECT_FALSE(parse_string<bool>("false"));
    EXPECT_FALSE(parse_string<bool>("no"));
    EXPECT_FALSE(parse_string<bool>("off"));
}

TEST(ParseStringTest, ParsesBooleansWithCaseAndWhitespace) {
    EXPECT_TRUE(parse_string<bool>(" YES "));
    EXPECT_FALSE(parse_string<bool>("\toFf\n"));
}

TEST(ParseStringTest, ThrowsOnInvalidBoolean) {
    EXPECT_THROW(parse_string<bool>("not_a_bool"), std::invalid_argument);
}

TEST(GetCurrentNumaNodeIdTest, ReturnsValidNumaNodeId) {
    int numa_node_id = get_current_numa_node_id();
    EXPECT_GE(numa_node_id, 0);
#if RAPIDSMPF_HAVE_NUMA
    EXPECT_LE(numa_node_id, numa_max_node());
#endif
}
