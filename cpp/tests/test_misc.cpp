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
    EXPECT_EQ(safe_cast<int64_t>(123456789L), 123456789L);
    EXPECT_EQ(safe_cast<size_t>(size_t{999}), size_t{999});
    EXPECT_DOUBLE_EQ(safe_cast<double>(3.14), 3.14);
    EXPECT_FLOAT_EQ(safe_cast<float>(2.5f), 2.5f);
}

// Test signed to unsigned conversions
TEST(MiscTest, SafeCastSignedToUnsignedSuccess) {
    // Valid positive values should succeed
    EXPECT_EQ(safe_cast<unsigned>(0), 0u);
    EXPECT_EQ(safe_cast<unsigned>(42), 42u);
    EXPECT_EQ(safe_cast<unsigned>(INT_MAX), static_cast<unsigned>(INT_MAX));
    EXPECT_EQ(safe_cast<size_t>(100), size_t{100});
    EXPECT_EQ(safe_cast<uint64_t>(int64_t{12345}), uint64_t{12345});
}

TEST(MiscTest, SafeCastSignedToUnsignedNegativeFails) {
    // Negative values should throw
    EXPECT_THROW(safe_cast<unsigned>(-1), std::logic_error);
    EXPECT_THROW(safe_cast<unsigned>(-42), std::logic_error);
    EXPECT_THROW(safe_cast<unsigned>(INT_MIN), std::logic_error);
    EXPECT_THROW(safe_cast<size_t>(-1), std::logic_error);
    EXPECT_THROW(safe_cast<uint32_t>(int32_t{-5}), std::logic_error);
    EXPECT_THROW(safe_cast<uint64_t>(int64_t{-999}), std::logic_error);
}

// Test unsigned to signed conversions
TEST(MiscTest, SafeCastUnsignedToSignedSuccess) {
    // Values within signed range should succeed
    EXPECT_EQ(safe_cast<int>(0u), 0);
    EXPECT_EQ(safe_cast<int>(42u), 42);
    EXPECT_EQ(safe_cast<int>(unsigned{INT_MAX}), INT_MAX);
    EXPECT_EQ(safe_cast<int32_t>(uint32_t{1000}), int32_t{1000});
}

TEST(MiscTest, SafeCastUnsignedToSignedOverflowFails) {
    // Values larger than signed max should throw
    EXPECT_THROW(safe_cast<int>(UINT_MAX), std::logic_error);
    EXPECT_THROW(safe_cast<int>(unsigned{INT_MAX} + 1u), std::logic_error);
    EXPECT_THROW(safe_cast<int32_t>(uint32_t{0x80000000}), std::logic_error);
    EXPECT_THROW(safe_cast<int64_t>(UINT64_MAX), std::logic_error);
}

// Test narrowing conversions (larger to smaller types)
TEST(MiscTest, SafeCastNarrowingSuccess) {
    // Values within target range should succeed
    EXPECT_EQ(safe_cast<int16_t>(int32_t{100}), int16_t{100});
    EXPECT_EQ(safe_cast<int16_t>(int32_t{-100}), int16_t{-100});
    EXPECT_EQ(safe_cast<uint8_t>(uint32_t{255}), uint8_t{255});
    EXPECT_EQ(safe_cast<int32_t>(int64_t{42}), int32_t{42});
    EXPECT_EQ(safe_cast<uint32_t>(uint64_t{100}), uint32_t{100});
}

TEST(MiscTest, SafeCastNarrowingOverflowFails) {
    // Values outside target range should throw
    EXPECT_THROW(safe_cast<int16_t>(int32_t{40000}), std::logic_error);
    EXPECT_THROW(safe_cast<int16_t>(int32_t{-40000}), std::logic_error);
    EXPECT_THROW(safe_cast<uint8_t>(uint32_t{256}), std::logic_error);
    EXPECT_THROW(safe_cast<int8_t>(int32_t{128}), std::logic_error);
    EXPECT_THROW(safe_cast<int8_t>(int32_t{-129}), std::logic_error);
    EXPECT_THROW(safe_cast<uint32_t>(uint64_t{0x100000000ULL}), std::logic_error);
}

// Test widening conversions (smaller to larger types - always safe)
TEST(MiscTest, SafeCastWideningConversions) {
    // All widening conversions should succeed
    EXPECT_EQ(safe_cast<int32_t>(int16_t{100}), int32_t{100});
    EXPECT_EQ(safe_cast<int32_t>(int16_t{-100}), int32_t{-100});
    EXPECT_EQ(safe_cast<int64_t>(int32_t{42}), int64_t{42});
    EXPECT_EQ(safe_cast<int64_t>(int32_t{-42}), int64_t{-42});
    EXPECT_EQ(safe_cast<uint32_t>(uint8_t{255}), uint32_t{255});
    EXPECT_EQ(safe_cast<uint64_t>(uint32_t{12345}), uint64_t{12345});
}

// Test edge cases with numeric limits
TEST(MiscTest, SafeCastNumericLimitsInt32) {
    // int32_t limits
    EXPECT_EQ(safe_cast<int32_t>(INT32_MIN), INT32_MIN);
    EXPECT_EQ(safe_cast<int32_t>(INT32_MAX), INT32_MAX);
    EXPECT_EQ(safe_cast<int64_t>(INT32_MIN), int64_t{INT32_MIN});
    EXPECT_EQ(safe_cast<int64_t>(INT32_MAX), int64_t{INT32_MAX});

    // uint32_t limits
    EXPECT_EQ(safe_cast<uint32_t>(0u), 0u);
    EXPECT_EQ(safe_cast<uint32_t>(UINT32_MAX), UINT32_MAX);
    EXPECT_EQ(safe_cast<uint64_t>(UINT32_MAX), uint64_t{UINT32_MAX});
    EXPECT_THROW(safe_cast<int32_t>(UINT32_MAX), std::logic_error);
}

TEST(MiscTest, SafeCastNumericLimitsInt64) {
    // int64_t limits
    EXPECT_EQ(safe_cast<int64_t>(INT64_MIN), INT64_MIN);
    EXPECT_EQ(safe_cast<int64_t>(INT64_MAX), INT64_MAX);
    EXPECT_THROW(safe_cast<int32_t>(INT64_MIN), std::logic_error);
    EXPECT_THROW(safe_cast<int32_t>(INT64_MAX), std::logic_error);

    // uint64_t limits
    EXPECT_EQ(safe_cast<uint64_t>(0ULL), 0ULL);
    EXPECT_EQ(safe_cast<uint64_t>(UINT64_MAX), UINT64_MAX);
    EXPECT_THROW(safe_cast<int64_t>(UINT64_MAX), std::logic_error);
}

// Test common use cases: ptrdiff_t conversions
TEST(MiscTest, SafeCastPtrdiffConversions) {
    // uint32_t to ptrdiff_t (should always succeed on 64-bit)
    EXPECT_EQ(safe_cast<std::ptrdiff_t>(uint32_t{0}), std::ptrdiff_t{0});
    EXPECT_EQ(safe_cast<std::ptrdiff_t>(uint32_t{100}), std::ptrdiff_t{100});
    EXPECT_EQ(safe_cast<std::ptrdiff_t>(UINT32_MAX), std::ptrdiff_t{UINT32_MAX});

    // uint64_t to ptrdiff_t (may overflow on 64-bit systems)
    EXPECT_EQ(safe_cast<std::ptrdiff_t>(uint64_t{0}), std::ptrdiff_t{0});
    EXPECT_EQ(safe_cast<std::ptrdiff_t>(uint64_t{100}), std::ptrdiff_t{100});

    // This should succeed on 64-bit, but we test both possibilities
    constexpr uint64_t large_value = uint64_t{1} << 62;
    if (large_value <= static_cast<uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()))
    {
        EXPECT_EQ(safe_cast<std::ptrdiff_t>(large_value), std::ptrdiff_t{large_value});
    } else {
        EXPECT_THROW(safe_cast<std::ptrdiff_t>(large_value), std::logic_error);
    }
}

// Test size_t conversions (common for array indexing)
TEST(MiscTest, SafeCastSizeTConversions) {
    // int to size_t
    EXPECT_EQ(safe_cast<size_t>(0), size_t{0});
    EXPECT_EQ(safe_cast<size_t>(42), size_t{42});
    EXPECT_EQ(safe_cast<size_t>(INT_MAX), static_cast<size_t>(INT_MAX));
    EXPECT_THROW(safe_cast<size_t>(-1), std::logic_error);
    EXPECT_THROW(safe_cast<size_t>(-100), std::logic_error);

    // size_t to int
    EXPECT_EQ(safe_cast<int>(size_t{0}), 0);
    EXPECT_EQ(safe_cast<int>(size_t{42}), 42);
    EXPECT_EQ(safe_cast<int>(size_t{INT_MAX}), INT_MAX);

    // Large size_t values should fail
    if (std::numeric_limits<size_t>::max() > static_cast<size_t>(INT_MAX)) {
        EXPECT_THROW(safe_cast<int>(size_t{INT_MAX} + 1), std::logic_error);
        EXPECT_THROW(
            safe_cast<int>(std::numeric_limits<size_t>::max()), std::logic_error
        );
    }
}

// Test MPI rank conversions (int32_t to size_t)
TEST(MiscTest, SafeCastRankConversions) {
    using Rank = int32_t;

    // Valid ranks
    EXPECT_EQ(safe_cast<size_t>(Rank{0}), size_t{0});
    EXPECT_EQ(safe_cast<size_t>(Rank{1}), size_t{1});
    EXPECT_EQ(safe_cast<size_t>(Rank{100}), size_t{100});
    EXPECT_EQ(safe_cast<size_t>(INT32_MAX), static_cast<size_t>(INT32_MAX));

    // Invalid ranks (negative)
    EXPECT_THROW(safe_cast<size_t>(Rank{-1}), std::logic_error);
    EXPECT_THROW(safe_cast<size_t>(INT32_MIN), std::logic_error);
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
        FAIL() << "Expected std::logic_error";
    } catch (const std::logic_error& e) {
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
    EXPECT_EQ(safe_cast<size_t>(0), size_t{0});
    EXPECT_EQ(safe_cast<int64_t>(0), int64_t{0});

    // Boundaries for int8_t
    EXPECT_EQ(safe_cast<int8_t>(int16_t{127}), int8_t{127});
    EXPECT_EQ(safe_cast<int8_t>(int16_t{-128}), int8_t{-128});
    EXPECT_THROW(safe_cast<int8_t>(int16_t{128}), std::logic_error);
    EXPECT_THROW(safe_cast<int8_t>(int16_t{-129}), std::logic_error);

    // Boundaries for uint8_t
    EXPECT_EQ(safe_cast<uint8_t>(uint16_t{255}), uint8_t{255});
    EXPECT_THROW(safe_cast<uint8_t>(uint16_t{256}), std::logic_error);
}

// Test mixed signed/unsigned of different sizes
TEST(MiscTest, SafeCastMixedSignednessAndSize) {
    // int64_t to uint32_t
    EXPECT_EQ(safe_cast<uint32_t>(int64_t{0}), uint32_t{0});
    EXPECT_EQ(safe_cast<uint32_t>(int64_t{100}), uint32_t{100});
    EXPECT_EQ(safe_cast<uint32_t>(int64_t{UINT32_MAX}), UINT32_MAX);
    EXPECT_THROW(safe_cast<uint32_t>(int64_t{-1}), std::logic_error);
    EXPECT_THROW(safe_cast<uint32_t>(int64_t{UINT32_MAX} + 1), std::logic_error);

    // uint64_t to int32_t
    EXPECT_EQ(safe_cast<int32_t>(uint64_t{0}), int32_t{0});
    EXPECT_EQ(safe_cast<int32_t>(uint64_t{100}), int32_t{100});
    EXPECT_EQ(safe_cast<int32_t>(uint64_t{INT32_MAX}), INT32_MAX);
    EXPECT_THROW(safe_cast<int32_t>(uint64_t{INT32_MAX} + 1), std::logic_error);
    EXPECT_THROW(safe_cast<int32_t>(UINT64_MAX), std::logic_error);
}
