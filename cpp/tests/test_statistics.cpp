/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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

    auto custom_formatter = [](std::ostream& os, std::size_t count, double val) {
        os << val << " by custom formatter";
    };

    EXPECT_EQ(stats.add_stat("custom-formatter", 10, custom_formatter), 10);
    EXPECT_EQ(stats.add_stat("custom-formatter", 1, custom_formatter), 11);
    EXPECT_EQ(stats.get_stat("custom-formatter").count_, 2);
    EXPECT_EQ(stats.get_stat("custom-formatter").value_, 11);
    EXPECT_THAT(stats.report(), ::testing::HasSubstr("custom-formatter"));
    EXPECT_THAT(stats.report(), ::testing::HasSubstr("11 by custom formatter"));

    EXPECT_EQ(stats.add_bytes_stat("byte-statistics", 20), 20);
    EXPECT_THAT(stats.report(), ::testing::HasSubstr("byte-statistics"));
    EXPECT_THAT(stats.report(), ::testing::HasSubstr("20.00 B"));
}
