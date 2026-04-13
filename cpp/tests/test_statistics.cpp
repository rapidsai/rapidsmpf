/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */


#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cudf/utilities/memory_resource.hpp>

#include <rapidsmpf/communicator/mpi.hpp>
#include <rapidsmpf/rmm_resource_adaptor.hpp>
#include <rapidsmpf/statistics.hpp>
#include <rapidsmpf/utils/string.hpp>

#include "environment.hpp"
#include "utils.hpp"

using namespace rapidsmpf;

/// @brief Test fixture that skips all tests on non-zero MPI ranks.
///
/// Statistics tests are all rank-independent.
class StatisticsTest : public ::testing::Test {
  protected:
    void SetUp() override {
        if (GlobalEnvironment->comm_->rank() != 0) {
            GTEST_SKIP() << "Test only runs on rank 0";
        }
    }
};

TEST_F(StatisticsTest, Disabled) {
    rapidsmpf::Statistics stats(false);
    EXPECT_FALSE(stats.enabled());

    // Disabed statistics is a no-op.
    stats.add_bytes_stat("name", 1);
    EXPECT_THROW(stats.get_stat("name"), std::out_of_range);
    EXPECT_THAT(stats.report(), ::testing::HasSubstr("Statistics: disabled"));
}

TEST_F(StatisticsTest, Communication) {
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

TEST_F(StatisticsTest, StatMax) {
    Statistics::Stat s;
    EXPECT_EQ(s.max(), -std::numeric_limits<double>::infinity());

    s.add(5.0);
    EXPECT_EQ(s.max(), 5.0);

    s.add(10.0);
    EXPECT_EQ(s.max(), 10.0);

    s.add(3.0);
    EXPECT_EQ(s.max(), 10.0);  // max stays at 10
}

TEST_F(StatisticsTest, ExistReportEntryName) {
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

TEST_F(StatisticsTest, RegisterFormatterFirstWins) {
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

TEST_F(StatisticsTest, MultiStatFormatter) {
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

TEST_F(StatisticsTest, ReportNoDataCollected) {
    rapidsmpf::Statistics stats;
    stats.register_formatter(
        "spill-summary",
        {"spill-bytes", "spill-time"},
        [](std::ostream& os, std::vector<rapidsmpf::Statistics::Stat> const& s) {
            os << format_nbytes(s[0].value()) << " in " << format_duration(s[1].value());
        }
    );
    // No stats recorded — formatter should still appear with "No data collected".
    EXPECT_THAT(stats.report(), ::testing::HasSubstr("spill-summary"));
    EXPECT_THAT(stats.report(), ::testing::HasSubstr("No data collected"));

    // Adding only one of the two required stats still yields "No data collected".
    stats.add_stat("spill-bytes", 1024 * 1024);
    EXPECT_THAT(stats.report(), ::testing::HasSubstr("No data collected"));
    EXPECT_THAT(stats.report(), ::testing::HasSubstr("spill-bytes"));  // uncovered
}

TEST_F(StatisticsTest, ReportSorting) {
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

TEST_F(StatisticsTest, MemoryProfiler) {
    rapidsmpf::RmmResourceAdaptor mr{cudf::get_current_device_resource_ref()};
    auto pinned_mr = rapidsmpf::PinnedMemoryResource::make_if_available();
    rapidsmpf::Statistics stats(&mr, pinned_mr);
    auto stream = cudf::get_default_stream();

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

        // pinned host memory
        if (pinned_mr != PinnedMemoryResource::Disabled) {
            void* ptr3 = pinned_mr->allocate(stream, 1_MiB);  // +1 MiB
            void* ptr4 = pinned_mr->allocate(stream, 2_MiB);  // +2 MiB
            pinned_mr->deallocate(stream, ptr3, 1_MiB);  // -1 MiB
            ptr3 = pinned_mr->allocate(stream, 1_MiB);  // +1 MiB
            pinned_mr->deallocate(stream, ptr4, 2_MiB);  // -2 MiB
            pinned_mr->deallocate(stream, ptr3, 1_MiB);  // -1 MiB
        }
        stream.synchronize();
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

    auto const report = stats.report();

    // Split the report on newlines and find the "main" record line.
    std::string main_line, pinned_line;
    {
        std::istringstream ss(report);
        std::string line;
        while (std::getline(ss, line) && (main_line.empty() || pinned_line.empty())) {
            if (line.find("main (all allocations using RmmResourceAdaptor)")
                != std::string::npos)
            {
                main_line = line;
            }
            if (line.find("main (all allocations using PinnedMemoryResource)")
                != std::string::npos)
            {
                pinned_line = line;
            }
        }
    }
    ASSERT_FALSE(main_line.empty()) << "main record line not found in report";
    ASSERT_FALSE(pinned_mr && pinned_line.empty())
        << "pinned record line found in report";

    // The report format is (statistics.cpp, lines 319-322):
    //   setw(8):num_calls  setw(12):peak  setw(12):g-peak  setw(12):accum  "  " name
    // For the main record: num_calls=1, peak=2 MiB, g-peak=2 MiB, accum=4 MiB.
    static constexpr std::string_view kExpectedMainLine =
        "       1       2 MiB       2 MiB       4 MiB       1 MiB"
        "  main (all allocations using RmmResourceAdaptor)";
    EXPECT_EQ(main_line, kExpectedMainLine);
    static const std::string_view kExpectedPinnedLine =
        pinned_mr == PinnedMemoryResource::Disabled
            ? ""
            : "       1       3 MiB       3 MiB       4 MiB       2 MiB"
              "  main (all allocations using PinnedMemoryResource)";
    EXPECT_EQ(pinned_line, kExpectedPinnedLine);
}

TEST_F(StatisticsTest, MemoryProfilerDisabled) {
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

TEST_F(StatisticsTest, MemoryProfilerMacro) {
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

TEST_F(StatisticsTest, MemoryProfilerMacroDisabled) {
    rapidsmpf::RmmResourceAdaptor mr{cudf::get_current_device_resource_ref()};
    rapidsmpf::Statistics stats(false);
    {
        RAPIDSMPF_MEMORY_PROFILE(stats);
        mr.deallocate_sync(mr.allocate_sync(1_MiB), 1_MiB);
    }
    auto const& records = stats.get_memory_records();
    EXPECT_TRUE(records.empty());
}

TEST_F(StatisticsTest, JsonStream) {
    rapidsmpf::Statistics stats;
    stats.add_stat("foo", 10.0);
    stats.add_stat("foo", 5.0);  // count=2, value=15, max=10
    stats.add_bytes_stat("bar", 1024);

    std::ostringstream ss;
    stats.write_json(ss);
    auto const& s = ss.str();

    EXPECT_THAT(s, ::testing::HasSubstr(R"("foo")"));
    EXPECT_THAT(s, ::testing::HasSubstr(R"("count": 2)"));
    EXPECT_THAT(s, ::testing::HasSubstr(R"("value": 15)"));
    EXPECT_THAT(s, ::testing::HasSubstr(R"("max": 10)"));
    EXPECT_THAT(s, ::testing::HasSubstr(R"("bar")"));
    EXPECT_THAT(s, ::testing::Not(::testing::HasSubstr("memory_records")));
}

TEST_F(StatisticsTest, InvalidStatNames) {
    rapidsmpf::Statistics stats;
    stats.add_stat("has\"quote", 1.0);
    stats.add_stat("has\\backslash", 2.0);
    std::ostringstream ss;
    EXPECT_THROW(stats.write_json(ss), std::invalid_argument);
}

TEST_F(StatisticsTest, InvalidMemoryRecordNames) {
    rapidsmpf::RmmResourceAdaptor mr{cudf::get_current_device_resource_ref()};
    rapidsmpf::Statistics stats(&mr);
    std::ignore = stats.create_memory_recorder("bad\"name");
    std::ostringstream ss;
    EXPECT_THROW(stats.write_json(ss), std::invalid_argument);
}

TEST_F(StatisticsTest, JsonMemoryRecords) {
    rapidsmpf::RmmResourceAdaptor mr{cudf::get_current_device_resource_ref()};
    rapidsmpf::Statistics stats(&mr);
    {
        auto rec = stats.create_memory_recorder("alloc");
        mr.deallocate_sync(mr.allocate_sync(1_MiB), 1_MiB);
    }

    std::ostringstream ss;
    stats.write_json(ss);
    auto const& s = ss.str();

    EXPECT_THAT(s, ::testing::HasSubstr("memory_records"));
    EXPECT_THAT(s, ::testing::HasSubstr(R"("alloc")"));
    EXPECT_THAT(s, ::testing::HasSubstr(R"("num_calls": 1)"));
    EXPECT_THAT(s, ::testing::HasSubstr(R"("peak_bytes")"));
    EXPECT_THAT(s, ::testing::HasSubstr(R"("total_bytes")"));
    EXPECT_THAT(s, ::testing::HasSubstr(R"("global_peak_bytes")"));
}

TEST_F(StatisticsTest, JsonReport) {
    rapidsmpf::Statistics stats;
    stats.add_stat("foo", 10.0);
    stats.add_stat("foo", 5.0);  // count=2, value=15, max=10
    stats.add_bytes_stat("bar", 1024);

    TempDir tmp_dir;
    auto const path = tmp_dir.path() / "stats.json";
    stats.write_json(path);

    std::ifstream f(path);
    ASSERT_TRUE(f.is_open());
    std::string file_contents(
        (std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>()
    );

    std::ostringstream ss;
    stats.write_json(ss);
    EXPECT_EQ(file_contents, ss.str());
}

TEST_F(StatisticsTest, StatConstructor) {
    Statistics::Stat s(5, 42.0, 10.0);
    EXPECT_EQ(s.count(), 5);
    EXPECT_EQ(s.value(), 42.0);
    EXPECT_EQ(s.max(), 10.0);
}

TEST_F(StatisticsTest, SerializeRoundTrip) {
    rapidsmpf::Statistics stats;
    stats.add_stat("alpha", 10.0);
    stats.add_stat("alpha", 5.0);  // count=2, value=15, max=10
    stats.add_stat("beta", 3.0);

    auto const bytes = stats.serialize();
    auto deserialized = rapidsmpf::Statistics::deserialize(bytes);

    EXPECT_TRUE(deserialized->enabled());
    EXPECT_EQ(deserialized->get_stat("alpha"), stats.get_stat("alpha"));
    EXPECT_EQ(deserialized->get_stat("beta"), stats.get_stat("beta"));
    EXPECT_EQ(deserialized->list_stat_names().size(), 2);
}

TEST_F(StatisticsTest, SerializeEmpty) {
    rapidsmpf::Statistics stats;
    auto const bytes = stats.serialize();
    auto deserialized = rapidsmpf::Statistics::deserialize(bytes);

    EXPECT_TRUE(deserialized->enabled());
    EXPECT_TRUE(deserialized->list_stat_names().empty());
}

TEST_F(StatisticsTest, SerializeMalformed) {
    // Empty data.
    std::vector<std::uint8_t> empty;
    EXPECT_THROW(
        std::ignore = rapidsmpf::Statistics::deserialize(empty), std::invalid_argument
    );

    // Truncated: just one byte, not enough for num_stats.
    std::vector<std::uint8_t> truncated = {1};
    EXPECT_THROW(
        std::ignore = rapidsmpf::Statistics::deserialize(truncated), std::invalid_argument
    );
}

TEST_F(StatisticsTest, Copy) {
    rapidsmpf::Statistics stats;
    stats.add_stat("x", 10.0);
    stats.register_formatter(
        "x", [](std::ostream& os, std::vector<rapidsmpf::Statistics::Stat> const& s) {
            os << "fmt:" << s[0].value();
        }
    );

    auto copied = stats.copy();
    EXPECT_TRUE(copied->enabled());
    EXPECT_EQ(copied->get_stat("x"), stats.get_stat("x"));
    EXPECT_THAT(copied->report(), ::testing::HasSubstr("fmt:10"));
}

TEST_F(StatisticsTest, MergeOverlapping) {
    auto a = std::make_shared<rapidsmpf::Statistics>();
    a->add_stat("x", 10.0);
    a->add_stat("x", 3.0);  // count=2, value=13, max=10

    auto b = std::make_shared<rapidsmpf::Statistics>();
    b->add_stat("x", 7.0);  // count=1, value=7, max=7

    auto merged = a->merge(b);
    auto s = merged->get_stat("x");
    EXPECT_EQ(s.count(), 3);
    EXPECT_EQ(s.value(), 20.0);
    EXPECT_EQ(s.max(), 10.0);
}

TEST_F(StatisticsTest, MergeDisjoint) {
    auto a = std::make_shared<rapidsmpf::Statistics>();
    a->add_stat("x", 1.0);

    auto b = std::make_shared<rapidsmpf::Statistics>();
    b->add_stat("y", 2.0);

    auto merged = a->merge(b);
    EXPECT_EQ(merged->list_stat_names().size(), 2);
    EXPECT_EQ(merged->get_stat("x"), a->get_stat("x"));
    EXPECT_EQ(merged->get_stat("y"), b->get_stat("y"));
}

TEST_F(StatisticsTest, MergeEmpty) {
    auto a = std::make_shared<rapidsmpf::Statistics>();
    a->add_stat("x", 5.0);

    auto empty = std::make_shared<rapidsmpf::Statistics>();

    auto merged = a->merge(empty);
    EXPECT_EQ(merged->get_stat("x"), a->get_stat("x"));
    EXPECT_EQ(merged->list_stat_names().size(), 1);

    auto merged2 = empty->merge(a);
    EXPECT_EQ(merged2->get_stat("x"), a->get_stat("x"));
    EXPECT_EQ(merged2->list_stat_names().size(), 1);
}

TEST_F(StatisticsTest, MergeUsesThisFormatters) {
    auto a = std::make_shared<rapidsmpf::Statistics>();
    a->register_formatter(
        "x", [](std::ostream& os, std::vector<rapidsmpf::Statistics::Stat> const& s) {
            os << "custom:" << s[0].value();
        }
    );
    a->add_stat("x", 10.0);

    auto b = std::make_shared<rapidsmpf::Statistics>();
    b->add_stat("x", 5.0);

    // Merging a (has formatter) with b: result uses a's formatter.
    auto merged = a->merge(b);
    EXPECT_THAT(merged->report(), ::testing::HasSubstr("custom:15"));

    // Merging b (no formatter) with a: result does not have the formatter.
    auto merged2 = b->merge(a);
    EXPECT_THAT(merged2->report(), ::testing::Not(::testing::HasSubstr("custom:")));
}

TEST_F(StatisticsTest, MergeSpan) {
    auto a = std::make_shared<rapidsmpf::Statistics>();
    a->add_stat("x", 1.0);

    auto b = std::make_shared<rapidsmpf::Statistics>();
    b->add_stat("x", 2.0);

    auto c = std::make_shared<rapidsmpf::Statistics>();
    c->add_stat("y", 10.0);

    std::vector<std::shared_ptr<rapidsmpf::Statistics>> others{b, c};
    auto merged = a->merge(std::span{others});

    EXPECT_EQ(merged->list_stat_names().size(), 2);
    EXPECT_EQ(merged->get_stat("x").value(), 3.0);
    EXPECT_EQ(merged->get_stat("y").value(), 10.0);
}
