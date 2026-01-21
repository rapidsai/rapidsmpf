/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rapidsmpf/utils/string.hpp>

#include "utils.hpp"

using namespace rapidsmpf;

TEST(UtilsTest, FormatsByteCount) {
    EXPECT_EQ(format_nbytes(0, 2, TrimZeroFraction::NO), "0.00 B");
    EXPECT_EQ(format_nbytes(0, 2, TrimZeroFraction::YES), "0 B");
    EXPECT_EQ(format_nbytes(1, 2, TrimZeroFraction::NO), "1.00 B");
    EXPECT_EQ(format_nbytes(1, 2, TrimZeroFraction::YES), "1 B");
    EXPECT_EQ(format_nbytes(512, 1, TrimZeroFraction::NO), "512.0 B");
    EXPECT_EQ(format_nbytes(512, 1, TrimZeroFraction::YES), "512 B");
    EXPECT_EQ(format_nbytes(1023, 1, TrimZeroFraction::NO), "1023.0 B");
    EXPECT_EQ(format_nbytes(1023, 1, TrimZeroFraction::YES), "1023 B");
    EXPECT_EQ(format_nbytes(1_KiB, 2, TrimZeroFraction::NO), "1.00 KiB");
    EXPECT_EQ(format_nbytes(1_KiB, 2, TrimZeroFraction::YES), "1 KiB");
    EXPECT_EQ(format_nbytes(1_MiB, 2, TrimZeroFraction::NO), "1.00 MiB");
    EXPECT_EQ(format_nbytes(1_MiB, 2, TrimZeroFraction::YES), "1 MiB");
    EXPECT_EQ(format_nbytes(10_MiB, 1, TrimZeroFraction::NO), "10.0 MiB");
    EXPECT_EQ(format_nbytes(10_MiB, 1, TrimZeroFraction::YES), "10 MiB");
    EXPECT_EQ(format_nbytes(10_MiB + 1_KiB, 2, TrimZeroFraction::YES), "10 MiB");
    EXPECT_EQ(format_nbytes(10_MiB + 1_KiB, 3, TrimZeroFraction::NO), "10.001 MiB");
    EXPECT_EQ(format_nbytes(1_GiB, 2, TrimZeroFraction::NO), "1.00 GiB");
    EXPECT_EQ(format_nbytes(1_GiB, 2, TrimZeroFraction::YES), "1 GiB");
    EXPECT_EQ(format_nbytes(1536, 2, TrimZeroFraction::NO), "1.50 KiB");
    EXPECT_EQ(format_nbytes(1536, 2, TrimZeroFraction::YES), "1.50 KiB");
    EXPECT_EQ(format_nbytes(-1, 2, TrimZeroFraction::NO), "-1.00 B");
    EXPECT_EQ(format_nbytes(-1, 2, TrimZeroFraction::YES), "-1 B");
    EXPECT_EQ(format_nbytes(-1024, 2, TrimZeroFraction::NO), "-1.00 KiB");
    EXPECT_EQ(format_nbytes(-1024, 2, TrimZeroFraction::YES), "-1 KiB");
    EXPECT_EQ(format_nbytes(-10 * (1 << 20), 1, TrimZeroFraction::NO), "-10.0 MiB");
    EXPECT_EQ(format_nbytes(-10 * (1 << 20), 1, TrimZeroFraction::YES), "-10 MiB");
}

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
