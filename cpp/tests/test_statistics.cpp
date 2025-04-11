/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */


#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rapidsmp/communicator/mpi.hpp>
#include <rapidsmp/statistics.hpp>

using namespace rapidsmp;

TEST(Statistics, Disabled) {
    rapidsmp::Statistics stats(false);
    EXPECT_FALSE(stats.enabled());

    // Disabed statistics is a no-op.
    EXPECT_EQ(stats.add_bytes_stat("name", 1), 0);
    EXPECT_THROW(stats.get_stat("name"), std::out_of_range);
    EXPECT_THAT(stats.report(), ::testing::HasSubstr("Statistics: disabled"));
}

TEST(Statistics, Communication) {
    rapidsmp::Statistics stats;
    EXPECT_TRUE(stats.enabled());

    EXPECT_THROW(stats.get_stat("unknown-name"), std::out_of_range);

    auto custom_formatter = [](std::ostream& os, std::size_t /* count */, double val) {
        os << val << " by custom formatter";
    };

    EXPECT_EQ(stats.add_stat("custom-formatter", 10, custom_formatter), 10);
    EXPECT_EQ(stats.add_stat("custom-formatter", 1, custom_formatter), 11);
    EXPECT_EQ(stats.get_stat("custom-formatter").count(), 2);
    EXPECT_EQ(stats.get_stat("custom-formatter").value(), 11);
    EXPECT_THAT(stats.report(), ::testing::HasSubstr("custom-formatter"));
    EXPECT_THAT(stats.report(), ::testing::HasSubstr("11 by custom formatter"));

    EXPECT_EQ(stats.add_bytes_stat("byte-statistics", 20), 20);
    EXPECT_THAT(stats.report(), ::testing::HasSubstr("byte-statistics"));
    EXPECT_THAT(stats.report(), ::testing::HasSubstr("20.00 B"));
}
