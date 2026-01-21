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

TEST(UtilsTest, FormatsDuration) {
    EXPECT_EQ(format_duration(0.0, 2, TrimZeroFraction::NO), "0.00 s");
    EXPECT_EQ(format_duration(0.0, 2, TrimZeroFraction::YES), "0 s");
    EXPECT_EQ(format_duration(1.0, 2, TrimZeroFraction::NO), "1.00 s");
    EXPECT_EQ(format_duration(1.0, 2, TrimZeroFraction::YES), "1 s");
    EXPECT_EQ(format_duration(1.234, 2, TrimZeroFraction::NO), "1.23 s");
    EXPECT_EQ(format_duration(1.234, 2, TrimZeroFraction::YES), "1.23 s");
    EXPECT_EQ(format_duration(0.5, 2, TrimZeroFraction::NO), "500.00 ms");
    EXPECT_EQ(format_duration(0.5, 2, TrimZeroFraction::YES), "500 ms");
    EXPECT_EQ(format_duration(0.001, 2, TrimZeroFraction::NO), "1.00 ms");
    EXPECT_EQ(format_duration(0.001, 2, TrimZeroFraction::YES), "1 ms");
    EXPECT_EQ(format_duration(0.000001, 2, TrimZeroFraction::NO), "1.00 µs");
    EXPECT_EQ(format_duration(0.000001, 2, TrimZeroFraction::YES), "1 µs");
    EXPECT_EQ(format_duration(0.000000001, 2, TrimZeroFraction::NO), "1.00 ns");
    EXPECT_EQ(format_duration(0.000000001, 2, TrimZeroFraction::YES), "1 ns");
    EXPECT_EQ(format_duration(60.0, 2, TrimZeroFraction::NO), "1.00 min");
    EXPECT_EQ(format_duration(60.0, 2, TrimZeroFraction::YES), "1 min");
    EXPECT_EQ(format_duration(65.0, 2, TrimZeroFraction::NO), "1.08 min");
    EXPECT_EQ(format_duration(65.0, 2, TrimZeroFraction::YES), "1.08 min");
    EXPECT_EQ(format_duration(3600.0, 2, TrimZeroFraction::NO), "1.00 h");
    EXPECT_EQ(format_duration(3600.0, 2, TrimZeroFraction::YES), "1 h");
    EXPECT_EQ(format_duration(86400.0, 2, TrimZeroFraction::NO), "1.00 d");
    EXPECT_EQ(format_duration(86400.0, 2, TrimZeroFraction::YES), "1 d");
    EXPECT_EQ(format_duration(-1.0, 2, TrimZeroFraction::NO), "-1.00 s");
    EXPECT_EQ(format_duration(-1.0, 2, TrimZeroFraction::YES), "-1 s");
    EXPECT_EQ(format_duration(-0.5, 2, TrimZeroFraction::NO), "-500.00 ms");
    EXPECT_EQ(format_duration(-0.5, 2, TrimZeroFraction::YES), "-500 ms");
    EXPECT_EQ(format_duration(-60.0, 2, TrimZeroFraction::NO), "-1.00 min");
    EXPECT_EQ(format_duration(-60.0, 2, TrimZeroFraction::YES), "-1 min");
}

TEST(UtilsTest, ParseNBytes) {
    EXPECT_EQ(parse_nbytes("0"), 0);
    EXPECT_EQ(parse_nbytes("1024"), 1_KiB);
    EXPECT_EQ(parse_nbytes("  42  "), 42);
    EXPECT_EQ(parse_nbytes("-7"), -7);
    EXPECT_EQ(parse_nbytes("1 B"), 1);
    EXPECT_EQ(parse_nbytes("1B"), 1);
    EXPECT_EQ(parse_nbytes("-1 B"), -1);
    EXPECT_EQ(parse_nbytes("  -1B  "), -1);
    EXPECT_EQ(parse_nbytes("1 KiB"), 1_KiB);
    EXPECT_EQ(parse_nbytes("1KiB"), 1_KiB);
    EXPECT_EQ(parse_nbytes("2 MiB"), 2_MiB);
    EXPECT_EQ(parse_nbytes("3GiB"), 3_GiB);
    EXPECT_EQ(parse_nbytes("1 kib"), 1024);
    EXPECT_EQ(parse_nbytes("1 mib"), 1_MiB);
    EXPECT_EQ(parse_nbytes("1 gib"), 1_GiB);
    EXPECT_EQ(parse_nbytes("1 KB"), 1e3);
    EXPECT_EQ(parse_nbytes("1MB"), 1e6);
    EXPECT_EQ(parse_nbytes("2 GB"), 2e9);
    EXPECT_EQ(parse_nbytes("1 kb"), 1e3);
    EXPECT_EQ(parse_nbytes("1 mb"), 1e6);
    EXPECT_EQ(parse_nbytes("1 gb"), 1e9);
    EXPECT_EQ(parse_nbytes("1.50 KiB"), 1536);
    EXPECT_EQ(parse_nbytes("-1.50 KiB"), -1536);
    EXPECT_EQ(parse_nbytes("0.5 B"), 1);
    EXPECT_EQ(parse_nbytes("-0.5 B"), -1);
    EXPECT_EQ(parse_nbytes("1e3 B"), 1e3);
    EXPECT_EQ(parse_nbytes("1e-3 KB"), 1);
    EXPECT_EQ(parse_nbytes("1e-3 KiB"), 1);  // 1.024 bytes rounds to 1
    EXPECT_EQ(parse_nbytes("-1e3 B"), -1e3);
    EXPECT_THROW(parse_nbytes("1 K"), std::invalid_argument);
    EXPECT_THROW(parse_nbytes("1 Ki"), std::invalid_argument);
    EXPECT_THROW(parse_nbytes("1 KiBB"), std::invalid_argument);
    EXPECT_THROW(parse_nbytes("1 KBps"), std::invalid_argument);
    EXPECT_THROW(parse_nbytes("1 BB"), std::invalid_argument);
    EXPECT_THROW(parse_nbytes(""), std::invalid_argument);
    EXPECT_THROW(parse_nbytes("   "), std::invalid_argument);
    EXPECT_THROW(parse_nbytes("abc"), std::invalid_argument);
    EXPECT_THROW(parse_nbytes("1.2.3 KiB"), std::invalid_argument);
    EXPECT_THROW(parse_nbytes("1 KiB extra"), std::invalid_argument);
    EXPECT_THROW(parse_nbytes("--1 KiB"), std::invalid_argument);
    EXPECT_THROW(parse_nbytes("1e309 B"), std::out_of_range);
    EXPECT_THROW(parse_nbytes("nan B"), std::invalid_argument);
    EXPECT_THROW(parse_nbytes("inf B"), std::invalid_argument);
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
