/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */


#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cudf/utilities/memory_resource.hpp>

#include <rapidsmpf/communicator/mpi.hpp>
#include <rapidsmpf/rmm_resource_adaptor.hpp>
#include <rapidsmpf/statistics.hpp>

#include "utils.hpp"

using namespace rapidsmpf;

TEST(Statistics, Disabled) {
    rapidsmpf::Statistics stats(false);
    EXPECT_FALSE(stats.enabled());

    // Disabed statistics is a no-op.
    EXPECT_EQ(stats.add_bytes_stat("name", 1), 0);
    EXPECT_THROW(stats.get_stat("name"), std::out_of_range);
    EXPECT_THAT(stats.report(), ::testing::HasSubstr("Statistics: disabled"));
}

TEST(Statistics, Communication) {
    rapidsmpf::Statistics stats;
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

TEST(Statistics, MemoryProfiler) {
    rapidsmpf::RmmResourceAdaptor mr{cudf::get_current_device_resource_ref()};
    rapidsmpf::Statistics stats(&mr);

    // Outer scope
    {
        auto outer = stats.create_memory_recorder("outer");
        void* ptr1 = mr.allocate(1_MiB);  // +1 MiB
        void* ptr2 = mr.allocate(1_MiB);  // +2 MiB
        mr.deallocate(ptr1, 1_MiB);
        mr.deallocate(ptr2, 1_MiB);

        // Nested scope
        {
            auto inner = stats.create_memory_recorder("inner");
            void* ptr3 = mr.allocate(1_MiB);  // +1 MiB
            mr.deallocate(ptr3, 1_MiB);
        }
    }
    auto const& records = stats.get_memory_records();

    // Verify outer
    EXPECT_EQ(records.at("outer").num_calls, 1);
    EXPECT_EQ(records.at("outer").global_peak, 2_MiB);
    EXPECT_EQ(records.at("outer").scoped.peak(), 2_MiB);
    EXPECT_EQ(records.at("outer").scoped.total(), 3_MiB);

    // Verify inner
    EXPECT_EQ(records.at("inner").num_calls, 1);
    EXPECT_EQ(records.at("inner").global_peak, 1_MiB);
    EXPECT_EQ(records.at("inner").scoped.peak(), 1_MiB);
    EXPECT_EQ(records.at("inner").scoped.total(), 1_MiB);

    // We can call the same name multiple times.
    {
        auto outer = stats.create_memory_recorder("outer");
        mr.deallocate(mr.allocate(1_MiB), 1_MiB);
    }
    EXPECT_EQ(records.at("outer").num_calls, 2);
    EXPECT_EQ(records.at("outer").global_peak, 2_MiB);
    EXPECT_EQ(records.at("outer").scoped.peak(), 2_MiB);
    EXPECT_EQ(records.at("outer").scoped.total(), 4_MiB);
}

TEST(Statistics, MemoryProfilerDisabled) {
    rapidsmpf::RmmResourceAdaptor mr{cudf::get_current_device_resource_ref()};
    rapidsmpf::Statistics stats(false);
    EXPECT_FALSE(stats.is_memory_profiling_enabled());

    // Outer scope
    {
        auto outer = stats.create_memory_recorder("outer");
        void* ptr1 = mr.allocate(1_MiB);  // +1 MiB
        void* ptr2 = mr.allocate(1_MiB);  // +2 MiB
        mr.deallocate(ptr1, 1_MiB);
        mr.deallocate(ptr2, 1_MiB);

        // Nested scope
        {
            auto inner = stats.create_memory_recorder("inner");
            void* ptr3 = mr.allocate(1_MiB);  // +1 MiB
            mr.deallocate(ptr3, 1_MiB);
        }
    }
    auto const& records = stats.get_memory_records();
    EXPECT_TRUE(records.empty());
}

TEST(Statistics, MemoryProfilerMacro) {
    rapidsmpf::RmmResourceAdaptor mr{cudf::get_current_device_resource_ref()};
    rapidsmpf::Statistics stats(&mr);
    {
        RAPIDSMPF_MEMORY_PROFILE(stats);
        mr.deallocate(mr.allocate(1_MiB), 1_MiB);
    }
    auto const& records = stats.get_memory_records();
    ASSERT_EQ(records.size(), 1);
    auto const& entry = *records.begin();
    EXPECT_TRUE(entry.first.find("test_statistics.cpp") != std::string::npos);
    EXPECT_EQ(entry.second.num_calls, 1);
    EXPECT_EQ(entry.second.scoped.total(), 1_MiB);
}

TEST(Statistics, MemoryProfilerMacroDisabled) {
    rapidsmpf::RmmResourceAdaptor mr{cudf::get_current_device_resource_ref()};
    rapidsmpf::Statistics stats(false);
    {
        RAPIDSMPF_MEMORY_PROFILE(stats);
        mr.deallocate(mr.allocate(1_MiB), 1_MiB);
    }
    auto const& records = stats.get_memory_records();
    EXPECT_TRUE(records.empty());
}
