/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */


#include <limits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cudf/utilities/memory_resource.hpp>

#include <rapidsmpf/communicator/mpi.hpp>
#include <rapidsmpf/rmm_resource_adaptor.hpp>
#include <rapidsmpf/statistics.hpp>
#include <rapidsmpf/utils/string.hpp>

#include "utils.hpp"

using namespace rapidsmpf;

TEST(Statistics, Disabled) {
    rapidsmpf::Statistics stats(false);
    EXPECT_FALSE(stats.enabled());

    // Disabed statistics is a no-op.
    stats.add_bytes_stat("name", 1);
    EXPECT_THROW(stats.get_stat("name"), std::out_of_range);
    EXPECT_THAT(stats.report(), ::testing::HasSubstr("Statistics: disabled"));
}

TEST(Statistics, Communication) {
    rapidsmpf::Statistics stats;
    EXPECT_TRUE(stats.enabled());

    EXPECT_THROW(stats.get_stat("unknown-name"), std::out_of_range);

    auto custom_formatter = [](std::ostream& os,
                               std::vector<rapidsmpf::Statistics::Stat> const& s) {
        os << s[0].value() << " by custom formatter";
    };

    stats.register_formatter("custom-formatter", custom_formatter);
    stats.add_stat("custom-formatter", 10);
    stats.add_stat("custom-formatter", 1);
    EXPECT_EQ(stats.get_stat("custom-formatter").count(), 2);
    EXPECT_EQ(stats.get_stat("custom-formatter").value(), 11);
    EXPECT_THAT(stats.report(), ::testing::HasSubstr("custom-formatter"));
    EXPECT_THAT(stats.report(), ::testing::HasSubstr("11 by custom formatter"));

    stats.add_bytes_stat("byte-statistics", 20);
    EXPECT_THAT(stats.report(), ::testing::HasSubstr("byte-statistics"));
    EXPECT_THAT(stats.report(), ::testing::HasSubstr("20 B"));
}

TEST(Statistics, StatMax) {
    Statistics::Stat s;
    EXPECT_EQ(s.max(), -std::numeric_limits<double>::infinity());

    s.add(5.0);
    EXPECT_EQ(s.max(), 5.0);

    s.add(10.0);
    EXPECT_EQ(s.max(), 10.0);

    s.add(3.0);
    EXPECT_EQ(s.max(), 10.0);  // max stays at 10
}

TEST(Statistics, ExistReportEntryName) {
    rapidsmpf::Statistics stats;

    // Unknown name returns false.
    EXPECT_FALSE(stats.exist_report_entry_name("foo"));

    // Returns true after registration.
    stats.register_formatter(
        "foo", [](std::ostream& os, std::vector<rapidsmpf::Statistics::Stat> const& s) {
            os << s[0].value();
        }
    );
    EXPECT_TRUE(stats.exist_report_entry_name("foo"));

    // Unrelated name is still absent.
    EXPECT_FALSE(stats.exist_report_entry_name("bar"));

    // Disabled statistics always returns false (no formatters are ever registered).
    rapidsmpf::Statistics disabled(false);
    disabled.register_formatter(
        "foo", [](std::ostream& os, std::vector<rapidsmpf::Statistics::Stat> const& s) {
            os << s[0].value();
        }
    );
    EXPECT_FALSE(disabled.exist_report_entry_name("foo"));
}

TEST(Statistics, RegisterFormatterFirstWins) {
    rapidsmpf::Statistics stats;
    // Register a custom formatter first.
    stats.register_formatter(
        "my-bytes",
        [](std::ostream& os, std::vector<rapidsmpf::Statistics::Stat> const& s) {
            os << "custom:" << s[0].value();
        }
    );
    // add_bytes_stat tries to register a bytes formatter, but the custom one takes
    // precedence because the first registered formatter is always used.
    stats.add_bytes_stat("my-bytes", 1024);
    EXPECT_THAT(stats.report(), ::testing::HasSubstr("custom:1024"));
    EXPECT_THAT(stats.report(), ::testing::Not(::testing::HasSubstr("KiB")));
}

TEST(Statistics, MultiStatFormatter) {
    rapidsmpf::Statistics stats;
    stats.register_formatter(
        "spill-summary",
        {"spill-bytes", "spill-time"},
        [](std::ostream& os, std::vector<rapidsmpf::Statistics::Stat> const& s) {
            os << format_nbytes(s[0].value()) << " in " << format_duration(s[1].value());
        }
    );
    stats.add_stat("spill-bytes", 1024 * 1024);
    stats.add_stat("spill-time", 0.001);
    EXPECT_THAT(stats.report(), ::testing::HasSubstr("spill-summary"));
    EXPECT_THAT(stats.report(), ::testing::HasSubstr("1 MiB"));
    // The component stats should not appear as individual report entries.
    EXPECT_THAT(stats.report(), ::testing::Not(::testing::HasSubstr("spill-bytes")));
    EXPECT_THAT(stats.report(), ::testing::Not(::testing::HasSubstr("spill-time")));
}

TEST(Statistics, ReportSorting) {
    rapidsmpf::Statistics stats;

    // Register formatter entries for "banana" and "cherry".
    stats.register_formatter(
        "banana",
        [](std::ostream& os, std::vector<rapidsmpf::Statistics::Stat> const& s) {
            os << s[0].value();
        }
    );
    stats.add_stat("banana", 2);

    stats.register_formatter(
        "cherry",
        [](std::ostream& os, std::vector<rapidsmpf::Statistics::Stat> const& s) {
            os << s[0].value();
        }
    );
    stats.add_stat("cherry", 3);

    // Add uncovered raw stats for "apple" and "date".
    stats.add_stat("apple", 1);
    stats.add_stat("date", 4);

    // All four entries must appear.
    auto const r = stats.report();
    auto const pos_apple = r.find("apple");
    auto const pos_banana = r.find("banana");
    auto const pos_cherry = r.find("cherry");
    auto const pos_date = r.find("date");

    ASSERT_NE(pos_apple, std::string::npos);
    ASSERT_NE(pos_banana, std::string::npos);
    ASSERT_NE(pos_cherry, std::string::npos);
    ASSERT_NE(pos_date, std::string::npos);

    // They must appear in alphabetical order.
    EXPECT_LT(pos_apple, pos_banana);
    EXPECT_LT(pos_banana, pos_cherry);
    EXPECT_LT(pos_cherry, pos_date);
}

TEST(Statistics, MemoryProfiler) {
    rapidsmpf::RmmResourceAdaptor mr{cudf::get_current_device_resource_ref()};
    rapidsmpf::Statistics stats(&mr);

    // Outer scope
    {
        auto outer = stats.create_memory_recorder("outer");
        void* ptr1 = mr.allocate_sync(1_MiB);  // +1 MiB
        void* ptr2 = mr.allocate_sync(1_MiB);  // +2 MiB
        mr.deallocate_sync(ptr1, 1_MiB);
        mr.deallocate_sync(ptr2, 1_MiB);

        // Nested scope
        {
            auto inner = stats.create_memory_recorder("inner");
            void* ptr3 = mr.allocate_sync(1_MiB);  // +1 MiB
            mr.deallocate_sync(ptr3, 1_MiB);
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
        mr.deallocate_sync(mr.allocate_sync(1_MiB), 1_MiB);
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
        void* ptr1 = mr.allocate_sync(1_MiB);  // +1 MiB
        void* ptr2 = mr.allocate_sync(1_MiB);  // +2 MiB
        mr.deallocate_sync(ptr1, 1_MiB);
        mr.deallocate_sync(ptr2, 1_MiB);

        // Nested scope
        {
            auto inner = stats.create_memory_recorder("inner");
            void* ptr3 = mr.allocate_sync(1_MiB);  // +1 MiB
            mr.deallocate_sync(ptr3, 1_MiB);
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
        mr.deallocate_sync(mr.allocate_sync(1_MiB), 1_MiB);
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
        mr.deallocate_sync(mr.allocate_sync(1_MiB), 1_MiB);
    }
    auto const& records = stats.get_memory_records();
    EXPECT_TRUE(records.empty());
}
