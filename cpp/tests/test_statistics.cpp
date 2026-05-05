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

    // Default-formatted stat (no report entry needed).
    stats.add_stat("plain-stat", 10);
    stats.add_stat("plain-stat", 1);
    EXPECT_EQ(stats.get_stat("plain-stat").count(), 2);
    EXPECT_EQ(stats.get_stat("plain-stat").value(), 11);
    EXPECT_THAT(stats.report(), ::testing::HasSubstr("plain-stat"));
    EXPECT_THAT(stats.report(), ::testing::HasSubstr("11 (count 2)"));

    stats.add_bytes_stat("byte-statistics", 20);
    EXPECT_THAT(stats.report(), ::testing::HasSubstr("byte-statistics"));
    EXPECT_THAT(stats.report(), ::testing::HasSubstr("20 B"));
}

TEST_F(StatisticsTest, AddReportEntryArityMismatchThrowsOnRender) {
    rapidsmpf::Statistics stats;
    // MemoryThroughput expects 3 stats; passing one is accepted at registration but
    // fails when report() tries to render the entry.
    stats.add_report_entry(
        "bad", {"only-one"}, rapidsmpf::Statistics::Formatter::MemoryThroughput
    );
    stats.add_stat("only-one", 1.0);
    EXPECT_THROW(std::ignore = stats.report(), std::out_of_range);
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

TEST_F(StatisticsTest, AddReportEntryFirstWins) {
    rapidsmpf::Statistics stats;
    // The first add_report_entry wins: a Default (count-aware) entry stays
    // in place even after add_bytes_stat tries to upgrade it to Bytes.
    stats.add_report_entry(
        "my-bytes", {"my-bytes"}, rapidsmpf::Statistics::Formatter::Default
    );
    stats.add_bytes_stat("my-bytes", 1024);
    EXPECT_THAT(stats.report(), ::testing::HasSubstr("my-bytes"));
    EXPECT_THAT(stats.report(), ::testing::HasSubstr("1024"));
    EXPECT_THAT(stats.report(), ::testing::Not(::testing::HasSubstr("KiB")));
}

TEST_F(StatisticsTest, MultiStatReportEntry) {
    rapidsmpf::Statistics stats;
    // Build a MemoryThroughput-style 3-stat report entry.
    stats.add_report_entry(
        "copy-summary",
        {"copy-summary-bytes", "copy-summary-time", "copy-summary-stream-delay"},
        rapidsmpf::Statistics::Formatter::MemoryThroughput
    );
    stats.add_stat("copy-summary-bytes", 1024 * 1024);
    stats.add_stat("copy-summary-time", 0.001);
    stats.add_stat("copy-summary-stream-delay", 0.0001);
    EXPECT_THAT(stats.report(), ::testing::HasSubstr("copy-summary"));
    EXPECT_THAT(stats.report(), ::testing::HasSubstr("1 MiB"));
    // Component stats are consumed by the report entry and don't emit
    // their own lines.
    EXPECT_THAT(
        stats.report(), ::testing::Not(::testing::HasSubstr("copy-summary-bytes"))
    );
}

TEST_F(StatisticsTest, ReportNoDataCollected) {
    rapidsmpf::Statistics stats;
    stats.add_report_entry(
        "spill-summary",
        {"spill-bytes", "spill-time", "spill-delay"},
        rapidsmpf::Statistics::Formatter::MemoryThroughput
    );
    // No stats recorded — entry should still appear with "No data collected".
    EXPECT_THAT(stats.report(), ::testing::HasSubstr("spill-summary"));
    EXPECT_THAT(stats.report(), ::testing::HasSubstr("No data collected"));

    // Adding only one of the three required stats still yields "No data collected".
    stats.add_stat("spill-bytes", 1024 * 1024);
    EXPECT_THAT(stats.report(), ::testing::HasSubstr("No data collected"));
    EXPECT_THAT(stats.report(), ::testing::HasSubstr("spill-bytes"));  // uncovered
}

TEST_F(StatisticsTest, ReportSorting) {
    rapidsmpf::Statistics stats;

    stats.add_stat("banana", 2);
    stats.add_report_entry(
        "banana", {"banana"}, rapidsmpf::Statistics::Formatter::Default
    );

    stats.add_stat("cherry", 3);
    stats.add_report_entry(
        "cherry", {"cherry"}, rapidsmpf::Statistics::Formatter::Default
    );

    // Uncovered raw stats for "apple" and "date".
    stats.add_stat("apple", 1);
    stats.add_stat("date", 4);

    auto const r = stats.report();
    auto const pos_apple = r.find("apple");
    auto const pos_banana = r.find("banana");
    auto const pos_cherry = r.find("cherry");
    auto const pos_date = r.find("date");

    ASSERT_NE(pos_apple, std::string::npos);
    ASSERT_NE(pos_banana, std::string::npos);
    ASSERT_NE(pos_cherry, std::string::npos);
    ASSERT_NE(pos_date, std::string::npos);

    EXPECT_LT(pos_apple, pos_banana);
    EXPECT_LT(pos_banana, pos_cherry);
    EXPECT_LT(pos_cherry, pos_date);
}

TEST_F(StatisticsTest, MemoryProfiler) {
    rapidsmpf::RmmResourceAdaptor mr{cudf::get_current_device_resource_ref()};
    auto pinned_mr = rapidsmpf::PinnedMemoryResource::make_if_available();
    rapidsmpf::Statistics stats;
    auto stream = cudf::get_default_stream();

    // Outer scope
    {
        auto outer = stats.create_memory_recorder(mr, "outer");
        void* ptr1 = mr.allocate_sync(1_MiB);  // +1 MiB
        void* ptr2 = mr.allocate_sync(1_MiB);  // +2 MiB
        mr.deallocate_sync(ptr1, 1_MiB);
        mr.deallocate_sync(ptr2, 1_MiB);

        // Nested scope
        {
            auto inner = stats.create_memory_recorder(mr, "inner");
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
        auto outer = stats.create_memory_recorder(mr, "outer");
        mr.deallocate_sync(mr.allocate_sync(1_MiB), 1_MiB);
    }
    EXPECT_EQ(records.at("outer").num_calls, 2);
    EXPECT_EQ(records.at("outer").global_peak, 2_MiB);
    EXPECT_EQ(records.at("outer").scoped.peak(), 2_MiB);
    EXPECT_EQ(records.at("outer").scoped.total(), 4_MiB);

    auto const report = stats.report({.mr = mr, .pinned_mr = pinned_mr});

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

    // Outer scope — pass nullopt so the recorder is a no-op.
    {
        auto outer = stats.create_memory_recorder(std::nullopt, "outer");
        void* ptr1 = mr.allocate_sync(1_MiB);  // +1 MiB
        void* ptr2 = mr.allocate_sync(1_MiB);  // +2 MiB
        mr.deallocate_sync(ptr1, 1_MiB);
        mr.deallocate_sync(ptr2, 1_MiB);

        // Nested scope
        {
            auto inner = stats.create_memory_recorder(std::nullopt, "inner");
            void* ptr3 = mr.allocate_sync(1_MiB);  // +1 MiB
            mr.deallocate_sync(ptr3, 1_MiB);
        }
    }
    auto const& records = stats.get_memory_records();
    EXPECT_TRUE(records.empty());
}

TEST_F(StatisticsTest, MemoryProfilerMacro) {
    rapidsmpf::RmmResourceAdaptor mr{cudf::get_current_device_resource_ref()};
    rapidsmpf::Statistics stats;
    {
        RAPIDSMPF_MEMORY_PROFILE(stats, mr);
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
        RAPIDSMPF_MEMORY_PROFILE(stats, std::nullopt);
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
    // JSON carries numeric data only — no formatter/report-entry metadata.
    EXPECT_THAT(s, ::testing::Not(::testing::HasSubstr("report_entries")));
    EXPECT_THAT(s, ::testing::Not(::testing::HasSubstr("formatter")));
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
    rapidsmpf::Statistics stats;
    std::ignore = stats.create_memory_recorder(mr, "bad\"name");
    std::ostringstream ss;
    EXPECT_THROW(stats.write_json(ss), std::invalid_argument);
}

TEST_F(StatisticsTest, JsonMemoryRecords) {
    rapidsmpf::RmmResourceAdaptor mr{cudf::get_current_device_resource_ref()};
    rapidsmpf::Statistics stats;
    {
        auto rec = stats.create_memory_recorder(mr, "alloc");
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

    // Truncated: just one byte, not enough for the num_stats field.
    std::vector<std::uint8_t> truncated = {1};
    EXPECT_THROW(
        std::ignore = rapidsmpf::Statistics::deserialize(truncated), std::invalid_argument
    );
}

TEST_F(StatisticsTest, DeserializeRejectsOutOfRangeFormatter) {
    // Craft a payload with one report-entry whose formatter byte is out of
    // range. Layout: [enabled=1][num_stats=0][num_entries=1]
    //                [name_len=1]["x"][formatter=0xFF][num_stat_names=1]
    //                [sn_len=1]["y"]
    auto const poke_u64 = [](std::vector<std::uint8_t>& v, std::uint64_t x) {
        for (int i = 0; i < 8; ++i)
            v.push_back(static_cast<std::uint8_t>((x >> (i * 8)) & 0xFF));
    };
    std::vector<std::uint8_t> buf;
    buf.push_back(1);  // enabled
    poke_u64(buf, 0);  // num_stats
    poke_u64(buf, 1);  // num_entries
    poke_u64(buf, 1);  // name_len
    buf.push_back('x');
    buf.push_back(0xFF);  // formatter value well past Formatter::_Count
    poke_u64(buf, 1);  // num_stat_names
    poke_u64(buf, 1);  // sn_len
    buf.push_back('y');

    EXPECT_THROW(
        std::ignore = rapidsmpf::Statistics::deserialize(buf), std::invalid_argument
    );
}

TEST_F(StatisticsTest, SerializeRoundTripPreservesEnabledFlag) {
    // A disabled Statistics should come back disabled after a round-trip.
    rapidsmpf::Statistics disabled(false);
    auto const bytes = disabled.serialize();
    auto deserialized = rapidsmpf::Statistics::deserialize(bytes);
    EXPECT_FALSE(deserialized->enabled());

    // And an enabled one comes back enabled.
    rapidsmpf::Statistics enabled(true);
    auto const bytes2 = enabled.serialize();
    auto deserialized2 = rapidsmpf::Statistics::deserialize(bytes2);
    EXPECT_TRUE(deserialized2->enabled());
}

TEST_F(StatisticsTest, SerializeRoundTripWithReportEntries) {
    rapidsmpf::Statistics stats;
    stats.add_bytes_stat("alpha", 2048);  // Bytes entry
    stats.add_duration_stat("beta", rapidsmpf::Duration{0.005});  // Duration entry
    stats.add_report_entry(
        "copy",
        {"copy-bytes", "copy-time", "copy-delay"},
        rapidsmpf::Statistics::Formatter::MemoryThroughput
    );
    stats.add_stat("copy-bytes", 1024.0 * 1024.0);
    stats.add_stat("copy-time", 0.002);
    stats.add_stat("copy-delay", 0.00001);

    auto const bytes = stats.serialize();
    auto deserialized = rapidsmpf::Statistics::deserialize(bytes);

    // Stats round-trip numerically.
    EXPECT_EQ(deserialized->get_stat("alpha"), stats.get_stat("alpha"));
    EXPECT_EQ(deserialized->get_stat("beta"), stats.get_stat("beta"));
    EXPECT_EQ(deserialized->get_stat("copy-bytes"), stats.get_stat("copy-bytes"));

    // And crucially, formatter metadata is preserved: the deserialized
    // report renders formatted values, not raw numbers.
    auto const deser_report = deserialized->report();
    EXPECT_THAT(deser_report, ::testing::HasSubstr("2 KiB"));  // alpha via Bytes
    EXPECT_THAT(deser_report, ::testing::HasSubstr("1 MiB"));  // copy composite
}

TEST_F(StatisticsTest, Copy) {
    rapidsmpf::Statistics stats;
    stats.add_bytes_stat("x", 2048);  // registers a Bytes report entry

    auto copied = stats.copy();
    EXPECT_TRUE(copied->enabled());
    EXPECT_EQ(copied->get_stat("x"), stats.get_stat("x"));
    // The Bytes formatter carried over the copy.
    EXPECT_THAT(copied->report(), ::testing::HasSubstr("2 KiB"));
}

TEST_F(StatisticsTest, MergeOverlapping) {
    auto a = std::make_shared<rapidsmpf::Statistics>();
    a->add_stat("x", 10.0);
    a->add_stat("x", 3.0);  // count=2, value=13, max=10

    auto b = std::make_shared<rapidsmpf::Statistics>();
    b->add_stat("x", 7.0);  // count=1, value=7, max=7

    std::vector<std::shared_ptr<rapidsmpf::Statistics>> inputs{a, b};
    auto merged = rapidsmpf::Statistics::merge(std::span{inputs});
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

    std::vector<std::shared_ptr<rapidsmpf::Statistics>> inputs{a, b};
    auto merged = rapidsmpf::Statistics::merge(std::span{inputs});
    EXPECT_EQ(merged->list_stat_names().size(), 2);
    EXPECT_EQ(merged->get_stat("x"), a->get_stat("x"));
    EXPECT_EQ(merged->get_stat("y"), b->get_stat("y"));
}

TEST_F(StatisticsTest, MergeWithEmpty) {
    auto a = std::make_shared<rapidsmpf::Statistics>();
    a->add_stat("x", 5.0);

    auto empty = std::make_shared<rapidsmpf::Statistics>();

    std::vector<std::shared_ptr<rapidsmpf::Statistics>> inputs{a, empty};
    auto merged = rapidsmpf::Statistics::merge(std::span{inputs});
    EXPECT_EQ(merged->get_stat("x"), a->get_stat("x"));
    EXPECT_EQ(merged->list_stat_names().size(), 1);

    std::vector<std::shared_ptr<rapidsmpf::Statistics>> rev{empty, a};
    auto merged2 = rapidsmpf::Statistics::merge(std::span{rev});
    EXPECT_EQ(merged2->get_stat("x"), a->get_stat("x"));
    EXPECT_EQ(merged2->list_stat_names().size(), 1);
}

TEST_F(StatisticsTest, MergeCombinesReportEntries) {
    auto a = std::make_shared<rapidsmpf::Statistics>();
    a->add_bytes_stat("x", 10);  // Bytes report entry

    auto b = std::make_shared<rapidsmpf::Statistics>();
    b->add_stat("x", 5.0);  // no formatter on this side

    // Merging a (has Bytes entry) with b: result uses a's entry.
    std::vector<std::shared_ptr<rapidsmpf::Statistics>> inputs{a, b};
    auto merged = rapidsmpf::Statistics::merge(std::span{inputs});
    EXPECT_THAT(merged->report(), ::testing::HasSubstr("15 B"));

    // Order doesn't matter for filling in a missing entry.
    std::vector<std::shared_ptr<rapidsmpf::Statistics>> rev{b, a};
    auto merged2 = rapidsmpf::Statistics::merge(std::span{rev});
    EXPECT_THAT(merged2->report(), ::testing::HasSubstr("15 B"));
}

TEST_F(StatisticsTest, MergeMultiple) {
    auto a = std::make_shared<rapidsmpf::Statistics>();
    a->add_stat("x", 1.0);

    auto b = std::make_shared<rapidsmpf::Statistics>();
    b->add_stat("x", 2.0);

    auto c = std::make_shared<rapidsmpf::Statistics>();
    c->add_stat("y", 10.0);

    std::vector<std::shared_ptr<rapidsmpf::Statistics>> inputs{a, b, c};
    auto merged = rapidsmpf::Statistics::merge(std::span{inputs});

    EXPECT_EQ(merged->list_stat_names().size(), 2);
    EXPECT_EQ(merged->get_stat("x").value(), 3.0);
    EXPECT_EQ(merged->get_stat("y").value(), 10.0);
}

TEST_F(StatisticsTest, MergeRejectsEmptySpan) {
    std::vector<std::shared_ptr<rapidsmpf::Statistics>> empty;
    EXPECT_THROW(
        std::ignore = rapidsmpf::Statistics::merge(std::span{empty}),
        std::invalid_argument
    );
}

TEST_F(StatisticsTest, MergeRejectsNullElement) {
    auto a = std::make_shared<rapidsmpf::Statistics>();
    std::vector<std::shared_ptr<rapidsmpf::Statistics>> inputs{a, nullptr};
    EXPECT_THROW(
        std::ignore = rapidsmpf::Statistics::merge(std::span{inputs}),
        std::invalid_argument
    );
}

TEST_F(StatisticsTest, MergeRejectsConflictingFormatter) {
    auto a = std::make_shared<rapidsmpf::Statistics>();
    a->add_report_entry("x", {"x"}, rapidsmpf::Statistics::Formatter::Bytes);
    a->add_stat("x", 1.0);

    auto b = std::make_shared<rapidsmpf::Statistics>();
    b->add_report_entry("x", {"x"}, rapidsmpf::Statistics::Formatter::Duration);
    b->add_stat("x", 2.0);

    std::vector<std::shared_ptr<rapidsmpf::Statistics>> inputs{a, b};
    EXPECT_THROW(
        std::ignore = rapidsmpf::Statistics::merge(std::span{inputs}),
        std::invalid_argument
    );
}

TEST_F(StatisticsTest, MergeIdenticalReportEntries) {
    // Two inputs with the same report entry (same formatter + stat_names)
    // must merge cleanly — no conflict.
    auto a = std::make_shared<rapidsmpf::Statistics>();
    a->add_bytes_stat("x", 10);

    auto b = std::make_shared<rapidsmpf::Statistics>();
    b->add_bytes_stat("x", 20);

    std::vector<std::shared_ptr<rapidsmpf::Statistics>> inputs{a, b};
    auto merged = rapidsmpf::Statistics::merge(std::span{inputs});
    EXPECT_THAT(merged->report(), ::testing::HasSubstr("30 B"));
}

TEST_F(StatisticsTest, MergeEnabledFlagPropagates) {
    auto enabled = std::make_shared<rapidsmpf::Statistics>(true);
    auto disabled = std::make_shared<rapidsmpf::Statistics>(false);

    // disabled + disabled -> disabled.
    std::vector<std::shared_ptr<rapidsmpf::Statistics>> both_off{disabled, disabled};
    EXPECT_FALSE(rapidsmpf::Statistics::merge(std::span{both_off})->enabled());

    // disabled + enabled -> enabled.
    std::vector<std::shared_ptr<rapidsmpf::Statistics>> mixed{disabled, enabled};
    EXPECT_TRUE(rapidsmpf::Statistics::merge(std::span{mixed})->enabled());
}

TEST_F(StatisticsTest, MergeRejectsConflictingStatNames) {
    auto a = std::make_shared<rapidsmpf::Statistics>();
    a->add_report_entry(
        "copy", {"b1", "t1", "d1"}, rapidsmpf::Statistics::Formatter::MemoryThroughput
    );

    auto b = std::make_shared<rapidsmpf::Statistics>();
    b->add_report_entry(
        "copy", {"b2", "t2", "d2"}, rapidsmpf::Statistics::Formatter::MemoryThroughput
    );

    std::vector<std::shared_ptr<rapidsmpf::Statistics>> inputs{a, b};
    EXPECT_THROW(
        std::ignore = rapidsmpf::Statistics::merge(std::span{inputs}),
        std::invalid_argument
    );
}
