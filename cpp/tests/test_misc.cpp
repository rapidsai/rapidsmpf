/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <limits>

#include <gtest/gtest.h>

#include <rapidsmpf/utils/misc.hpp>

using namespace rapidsmpf;

// Test same-type conversions (should be no-op)
TEST(MiscTest, SafeCastSameTypeConversions) {
    EXPECT_EQ(safe_cast<int>(42), 42);
    EXPECT_EQ(safe_cast<unsigned>(100u), 100u);
    EXPECT_EQ(safe_cast<std::int64_t>(123456789L), 123456789L);
    EXPECT_EQ(safe_cast<std::size_t>(std::size_t{999}), std::size_t{999});
    EXPECT_DOUBLE_EQ(safe_cast<double>(3.14), 3.14);
    EXPECT_FLOAT_EQ(safe_cast<float>(2.5f), 2.5f);
}

// Test signed to unsigned conversions
TEST(MiscTest, SafeCastSignedToUnsignedSuccess) {
    // Valid positive values should succeed
    EXPECT_EQ(safe_cast<unsigned>(0), 0u);
    EXPECT_EQ(safe_cast<unsigned>(42), 42u);
    EXPECT_EQ(safe_cast<unsigned>(INT_MAX), static_cast<unsigned>(INT_MAX));
    EXPECT_EQ(safe_cast<std::size_t>(100), std::size_t{100});
    EXPECT_EQ(safe_cast<std::uint64_t>(std::int64_t{12345}), std::uint64_t{12345});
}

TEST(MiscTest, SafeCastSignedToUnsignedNegativeFails) {
    // Negative values should throw
    EXPECT_THROW(safe_cast<unsigned>(-1), std::overflow_error);
    EXPECT_THROW(safe_cast<unsigned>(-42), std::overflow_error);
    EXPECT_THROW(safe_cast<unsigned>(INT_MIN), std::overflow_error);
    EXPECT_THROW(safe_cast<std::size_t>(-1), std::overflow_error);
    EXPECT_THROW(safe_cast<std::uint32_t>(std::int32_t{-5}), std::overflow_error);
    EXPECT_THROW(safe_cast<std::uint64_t>(std::int64_t{-999}), std::overflow_error);
}

// Test unsigned to signed conversions
TEST(MiscTest, SafeCastUnsignedToSignedSuccess) {
    // Values within signed range should succeed
    EXPECT_EQ(safe_cast<int>(0u), 0);
    EXPECT_EQ(safe_cast<int>(42u), 42);
    EXPECT_EQ(safe_cast<int>(unsigned{INT_MAX}), INT_MAX);
    EXPECT_EQ(safe_cast<std::int32_t>(std::uint32_t{1000}), std::int32_t{1000});
}

TEST(MiscTest, SafeCastUnsignedToSignedOverflowFails) {
    // Values larger than signed max should throw
    EXPECT_THROW(safe_cast<int>(UINT_MAX), std::overflow_error);
    EXPECT_THROW(safe_cast<int>(unsigned{INT_MAX} + 1u), std::overflow_error);
    EXPECT_THROW(safe_cast<std::int32_t>(std::uint32_t{0x80000000}), std::overflow_error);
    EXPECT_THROW(safe_cast<std::int64_t>(UINT64_MAX), std::overflow_error);
}

// Test narrowing conversions (larger to smaller types)
TEST(MiscTest, SafeCastNarrowingSuccess) {
    // Values within target range should succeed
    EXPECT_EQ(safe_cast<std::int16_t>(std::int32_t{100}), std::int16_t{100});
    EXPECT_EQ(safe_cast<std::int16_t>(std::int32_t{-100}), std::int16_t{-100});
    EXPECT_EQ(safe_cast<std::uint8_t>(std::uint32_t{255}), std::uint8_t{255});
    EXPECT_EQ(safe_cast<std::int32_t>(std::int64_t{42}), std::int32_t{42});
    EXPECT_EQ(safe_cast<std::uint32_t>(std::uint64_t{100}), std::uint32_t{100});
}

TEST(MiscTest, SafeCastNarrowingOverflowFails) {
    // Values outside target range should throw
    EXPECT_THROW(safe_cast<std::int16_t>(std::int32_t{40000}), std::overflow_error);
    EXPECT_THROW(safe_cast<std::int16_t>(std::int32_t{-40000}), std::overflow_error);
    EXPECT_THROW(safe_cast<std::uint8_t>(std::uint32_t{256}), std::overflow_error);
    EXPECT_THROW(safe_cast<std::int8_t>(std::int32_t{128}), std::overflow_error);
    EXPECT_THROW(safe_cast<std::int8_t>(std::int32_t{-129}), std::overflow_error);
    EXPECT_THROW(
        safe_cast<std::uint32_t>(std::uint64_t{0x100000000ULL}), std::overflow_error
    );
}

// Test widening conversions (smaller to larger types - always safe)
TEST(MiscTest, SafeCastWideningConversions) {
    // All widening conversions should succeed
    EXPECT_EQ(safe_cast<std::int32_t>(std::int16_t{100}), std::int32_t{100});
    EXPECT_EQ(safe_cast<std::int32_t>(std::int16_t{-100}), std::int32_t{-100});
    EXPECT_EQ(safe_cast<std::int64_t>(std::int32_t{42}), std::int64_t{42});
    EXPECT_EQ(safe_cast<std::int64_t>(std::int32_t{-42}), std::int64_t{-42});
    EXPECT_EQ(safe_cast<std::uint32_t>(std::uint8_t{255}), std::uint32_t{255});
    EXPECT_EQ(safe_cast<std::uint64_t>(std::uint32_t{12345}), std::uint64_t{12345});
}

// Test edge cases with numeric limits
TEST(MiscTest, SafeCastNumericLimitsInt32) {
    // std::int32_t limits
    EXPECT_EQ(safe_cast<std::int32_t>(INT32_MIN), INT32_MIN);
    EXPECT_EQ(safe_cast<std::int32_t>(INT32_MAX), INT32_MAX);
    EXPECT_EQ(safe_cast<std::int64_t>(INT32_MIN), std::int64_t{INT32_MIN});
    EXPECT_EQ(safe_cast<std::int64_t>(INT32_MAX), std::int64_t{INT32_MAX});

    // std::uint32_t limits
    EXPECT_EQ(safe_cast<std::uint32_t>(0u), 0u);
    EXPECT_EQ(safe_cast<std::uint32_t>(UINT32_MAX), UINT32_MAX);
    EXPECT_EQ(safe_cast<std::uint64_t>(UINT32_MAX), std::uint64_t{UINT32_MAX});
    EXPECT_THROW(safe_cast<std::int32_t>(UINT32_MAX), std::overflow_error);
}

TEST(MiscTest, SafeCastNumericLimitsInt64) {
    // std::int64_t limits
    EXPECT_EQ(safe_cast<std::int64_t>(INT64_MIN), INT64_MIN);
    EXPECT_EQ(safe_cast<std::int64_t>(INT64_MAX), INT64_MAX);
    EXPECT_THROW(safe_cast<std::int32_t>(INT64_MIN), std::overflow_error);
    EXPECT_THROW(safe_cast<std::int32_t>(INT64_MAX), std::overflow_error);

    // std::uint64_t limits
    EXPECT_EQ(safe_cast<std::uint64_t>(0ULL), 0ULL);
    EXPECT_EQ(safe_cast<std::uint64_t>(UINT64_MAX), UINT64_MAX);
    EXPECT_THROW(safe_cast<std::int64_t>(UINT64_MAX), std::overflow_error);
}

// Test common use cases: ptrdiff_t conversions
TEST(MiscTest, SafeCastPtrdiffConversions) {
    // std::uint32_t to ptrdiff_t (should always succeed on 64-bit)
    EXPECT_EQ(safe_cast<std::ptrdiff_t>(std::uint32_t{0}), std::ptrdiff_t{0});
    EXPECT_EQ(safe_cast<std::ptrdiff_t>(std::uint32_t{100}), std::ptrdiff_t{100});
    EXPECT_EQ(safe_cast<std::ptrdiff_t>(UINT32_MAX), std::ptrdiff_t{UINT32_MAX});

    // std::uint64_t to ptrdiff_t (may overflow on 64-bit systems)
    EXPECT_EQ(safe_cast<std::ptrdiff_t>(std::uint64_t{0}), std::ptrdiff_t{0});
    EXPECT_EQ(safe_cast<std::ptrdiff_t>(std::uint64_t{100}), std::ptrdiff_t{100});

    // This should succeed on 64-bit, but we test both possibilities
    constexpr std::uint64_t large_value = std::uint64_t{1} << 62;
    if (large_value
        <= static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()))
    {
        EXPECT_EQ(safe_cast<std::ptrdiff_t>(large_value), std::ptrdiff_t{large_value});
    } else {
        EXPECT_THROW(safe_cast<std::ptrdiff_t>(large_value), std::overflow_error);
    }
}

// Test std::size_t conversions (common for array indexing)
TEST(MiscTest, SafeCastSizeTConversions) {
    // int to std::size_t
    EXPECT_EQ(safe_cast<std::size_t>(0), std::size_t{0});
    EXPECT_EQ(safe_cast<std::size_t>(42), std::size_t{42});
    EXPECT_EQ(safe_cast<std::size_t>(INT_MAX), static_cast<std::size_t>(INT_MAX));
    EXPECT_THROW(safe_cast<std::size_t>(-1), std::overflow_error);
    EXPECT_THROW(safe_cast<std::size_t>(-100), std::overflow_error);

    // std::size_t to int
    EXPECT_EQ(safe_cast<int>(std::size_t{0}), 0);
    EXPECT_EQ(safe_cast<int>(std::size_t{42}), 42);
    EXPECT_EQ(safe_cast<int>(std::size_t{INT_MAX}), INT_MAX);

    // Large std::size_t values should fail
    if (std::numeric_limits<std::size_t>::max() > static_cast<std::size_t>(INT_MAX)) {
        EXPECT_THROW(safe_cast<int>(std::size_t{INT_MAX} + 1), std::overflow_error);
        EXPECT_THROW(
            safe_cast<int>(std::numeric_limits<std::size_t>::max()), std::overflow_error
        );
    }
}

// Test MPI rank conversions (std::int32_t to std::size_t)
TEST(MiscTest, SafeCastRankConversions) {
    using Rank = std::int32_t;

    // Valid ranks
    EXPECT_EQ(safe_cast<std::size_t>(Rank{0}), std::size_t{0});
    EXPECT_EQ(safe_cast<std::size_t>(Rank{1}), std::size_t{1});
    EXPECT_EQ(safe_cast<std::size_t>(Rank{100}), std::size_t{100});
    EXPECT_EQ(safe_cast<std::size_t>(INT32_MAX), static_cast<std::size_t>(INT32_MAX));

    // Invalid ranks (negative)
    EXPECT_THROW(safe_cast<std::size_t>(Rank{-1}), std::overflow_error);
    EXPECT_THROW(safe_cast<std::size_t>(INT32_MIN), std::overflow_error);
}

// Test floating point conversions (should be direct cast)
TEST(MiscTest, SafeCastFloatingPointConversions) {
    // float to double
    EXPECT_DOUBLE_EQ(safe_cast<double>(1.5f), 1.5);
    EXPECT_DOUBLE_EQ(safe_cast<double>(-2.5f), -2.5);

    // double to float
    EXPECT_FLOAT_EQ(safe_cast<float>(3.14), 3.14f);
    EXPECT_FLOAT_EQ(safe_cast<float>(-1.5), -1.5f);

    // int to float/double (should just cast)
    EXPECT_FLOAT_EQ(safe_cast<float>(42), 42.0f);
    EXPECT_DOUBLE_EQ(safe_cast<double>(100), 100.0);

    // float/double to int (should just cast)
    EXPECT_EQ(safe_cast<int>(3.14), 3);
    EXPECT_EQ(safe_cast<int>(-2.7), -2);
}

// Test that error messages include source location
TEST(MiscTest, SafeCastErrorMessageContainsLocation) {
    try {
        safe_cast<unsigned>(-1);
        FAIL() << "Expected std::overflow_error";
    } catch (const std::overflow_error& e) {
        std::string msg(e.what());
        // Error message should contain file name and line number
        EXPECT_TRUE(msg.find("test_misc.cpp") != std::string::npos)
            << "Error message should contain file name: " << msg;
        EXPECT_TRUE(msg.find("RapidsMPF cast error") != std::string::npos)
            << "Error message should contain 'RapidsMPF cast error': " << msg;
    }
}

// Test zero and boundary values
TEST(MiscTest, SafeCastZeroAndBoundaries) {
    // Zero should always work
    EXPECT_EQ(safe_cast<int>(0), 0);
    EXPECT_EQ(safe_cast<unsigned>(0), 0u);
    EXPECT_EQ(safe_cast<std::size_t>(0), std::size_t{0});
    EXPECT_EQ(safe_cast<std::int64_t>(0), std::int64_t{0});

    // Boundaries for std::int8_t
    EXPECT_EQ(safe_cast<std::int8_t>(std::int16_t{127}), std::int8_t{127});
    EXPECT_EQ(safe_cast<std::int8_t>(std::int16_t{-128}), std::int8_t{-128});
    EXPECT_THROW(safe_cast<std::int8_t>(std::int16_t{128}), std::overflow_error);
    EXPECT_THROW(safe_cast<std::int8_t>(std::int16_t{-129}), std::overflow_error);

    // Boundaries for std::uint8_t
    EXPECT_EQ(safe_cast<std::uint8_t>(std::uint16_t{255}), std::uint8_t{255});
    EXPECT_THROW(safe_cast<std::uint8_t>(std::uint16_t{256}), std::overflow_error);
}

// Test mixed signed/unsigned of different sizes
TEST(MiscTest, SafeCastMixedSignednessAndSize) {
    // std::int64_t to std::uint32_t
    EXPECT_EQ(safe_cast<std::uint32_t>(std::int64_t{0}), std::uint32_t{0});
    EXPECT_EQ(safe_cast<std::uint32_t>(std::int64_t{100}), std::uint32_t{100});
    EXPECT_EQ(safe_cast<std::uint32_t>(std::int64_t{UINT32_MAX}), UINT32_MAX);
    EXPECT_THROW(safe_cast<std::uint32_t>(std::int64_t{-1}), std::overflow_error);
    EXPECT_THROW(
        safe_cast<std::uint32_t>(std::int64_t{UINT32_MAX} + 1), std::overflow_error
    );

    // std::uint64_t to std::int32_t
    EXPECT_EQ(safe_cast<std::int32_t>(std::uint64_t{0}), std::int32_t{0});
    EXPECT_EQ(safe_cast<std::int32_t>(std::uint64_t{100}), std::int32_t{100});
    EXPECT_EQ(safe_cast<std::int32_t>(std::uint64_t{INT32_MAX}), INT32_MAX);
    EXPECT_THROW(
        safe_cast<std::int32_t>(std::uint64_t{INT32_MAX} + 1), std::overflow_error
    );
    EXPECT_THROW(safe_cast<std::int32_t>(UINT64_MAX), std::overflow_error);
}
