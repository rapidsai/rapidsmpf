/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#ifdef RAPIDSMPF_HAVE_CUPTI
#include <rapidsmpf/cupti.hpp>
#endif

/**
 * @brief Integration tests for CUPTI support compilation and linking
 */

#ifdef RAPIDSMPF_HAVE_CUPTI

TEST(CuptiIntegration, HeaderInclude) {
    // Test that the header includes properly
    static_assert(std::is_default_constructible_v<rapidsmpf::CuptiMonitor>);
    static_assert(!std::is_copy_constructible_v<rapidsmpf::CuptiMonitor>);
    static_assert(!std::is_move_constructible_v<rapidsmpf::CuptiMonitor>);
}

TEST(CuptiIntegration, BasicInstantiation) {
    // Test that we can create a CuptiMonitor instance
    ASSERT_NO_THROW({
        rapidsmpf::CuptiMonitor monitor;
        EXPECT_FALSE(monitor.is_monitoring());
        EXPECT_EQ(monitor.get_sample_count(), 0);
    });
}

TEST(CuptiIntegration, MemoryDataPointStructure) {
    // Test that MemoryDataPoint structure is properly defined
    rapidsmpf::MemoryDataPoint point;
    point.timestamp = 1.0;
    point.free_memory = 1000;
    point.total_memory = 2000;
    point.used_memory = 1000;

    EXPECT_EQ(point.timestamp, 1.0);
    EXPECT_EQ(point.free_memory, 1000);
    EXPECT_EQ(point.total_memory, 2000);
    EXPECT_EQ(point.used_memory, 1000);
}

TEST(CuptiIntegration, ConditionalCompilation) {
// Test that RAPIDSMPF_HAVE_CUPTI is properly defined
#ifdef RAPIDSMPF_HAVE_CUPTI
    SUCCEED() << "CUPTI support is properly enabled";
#else
    FAIL() << "CUPTI support should be enabled in this compilation";
#endif
}

#else

TEST(CuptiIntegration, CuptiDisabled) {
    // Test runs when CUPTI is disabled
    GTEST_SKIP() << "CUPTI support is disabled. This is expected in builds without "
                    "-DBUILD_CUPTI_SUPPORT=ON";
}

#endif  // RAPIDSMPF_HAVE_CUPTI

// This test always runs regardless of CUPTI availability
TEST(CuptiIntegration, CompilationTest) {
// Test that the test file compiles properly in both cases
#ifdef RAPIDSMPF_HAVE_CUPTI
    // CUPTI is available
    EXPECT_TRUE(true) << "CUPTI support compiled successfully";
#else
    // CUPTI is not available
    EXPECT_TRUE(true) << "Code compiles correctly without CUPTI support";
#endif
}
