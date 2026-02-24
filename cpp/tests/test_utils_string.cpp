/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <sstream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rapidsmpf/memory/memory_type.hpp>
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
    EXPECT_EQ(format_duration(0.000001, 2, TrimZeroFraction::NO), "1.00 us");
    EXPECT_EQ(format_duration(0.000001, 2, TrimZeroFraction::YES), "1 us");
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

TEST(UtilsTest, ParseNBytesUnsigned) {
    EXPECT_EQ(parse_nbytes_unsigned("0"), 0u);
    EXPECT_EQ(parse_nbytes_unsigned("1024"), std::size_t{1_KiB});
    EXPECT_EQ(parse_nbytes_unsigned("1 KiB"), std::size_t{1_KiB});
    EXPECT_EQ(parse_nbytes_unsigned("1 KB"), std::size_t{1000});
    EXPECT_EQ(parse_nbytes_unsigned("1.50 KiB"), std::size_t{1536});
    EXPECT_EQ(
        parse_nbytes_unsigned("1e-3 KiB"), std::size_t{1}
    );  // 1.024 bytes rounds to 1

    EXPECT_THROW(parse_nbytes_unsigned("-1"), std::invalid_argument);
    EXPECT_THROW(parse_nbytes_unsigned("-1 KiB"), std::invalid_argument);
    EXPECT_THROW(parse_nbytes_unsigned("-0.5 B"), std::invalid_argument);
    EXPECT_THROW(parse_nbytes_unsigned("1 KBps"), std::invalid_argument);
    EXPECT_THROW(parse_nbytes_unsigned("abc"), std::invalid_argument);
    EXPECT_THROW(parse_nbytes_unsigned("1e309 B"), std::out_of_range);
}

TEST(UtilsTest, ParseNBytesOrPercent) {
    // Absolute byte quantity: total_bytes is ignored (but must be > 0)
    EXPECT_EQ(parse_nbytes_or_percent("0", 1), std::size_t{0});
    EXPECT_EQ(parse_nbytes_or_percent("1024", 1), std::size_t{1024});
    EXPECT_EQ(parse_nbytes_or_percent("1 KiB", 1), std::size_t{1_KiB});
    EXPECT_EQ(parse_nbytes_or_percent("1 KB", 1), std::size_t{1000});
    EXPECT_EQ(parse_nbytes_or_percent("  1.50 KiB  ", 1), std::size_t{1536});
    EXPECT_EQ(parse_nbytes_or_percent("1e-3 KiB", 1), std::size_t{1});  // rounds to 1

    // Percentage: percent of total_bytes, floored to bytes
    EXPECT_EQ(parse_nbytes_or_percent("0%", 2_KiB), std::size_t{0});
    EXPECT_EQ(parse_nbytes_or_percent("50%", 2_KiB), std::size_t{1_KiB});
    EXPECT_EQ(parse_nbytes_or_percent("  50 %  ", 2000), std::size_t{1000});

    // Note: "1.5%" -> parse_nbytes_unsigned("1.5") rounds to 2
    // floor((2 / 100) * 200) = 4
    EXPECT_EQ(parse_nbytes_or_percent("1.5%", 200), std::size_t{4});

    // Errors
    EXPECT_THROW(parse_nbytes_or_percent("-1", 1), std::invalid_argument);
    EXPECT_THROW(parse_nbytes_or_percent("-1%", 1), std::invalid_argument);
    EXPECT_THROW(parse_nbytes_or_percent("1 KBps", 1), std::invalid_argument);
    EXPECT_THROW(parse_nbytes_or_percent("abc", 1), std::invalid_argument);
    EXPECT_THROW(parse_nbytes_or_percent("1 GiB%", 1), std::invalid_argument);

    // total_bytes validation runs for all inputs
    EXPECT_THROW(parse_nbytes_or_percent("1%", 0), std::invalid_argument);
    EXPECT_THROW(parse_nbytes_or_percent("1", 0), std::invalid_argument);

    EXPECT_THROW(parse_nbytes_or_percent("1e309 B", 1), std::out_of_range);
    EXPECT_THROW(parse_nbytes_or_percent("1e309%", 1), std::out_of_range);
}

TEST(UtilsTest, ParseDuration) {
    EXPECT_DOUBLE_EQ(parse_duration("0").count(), 0.0);
    EXPECT_DOUBLE_EQ(parse_duration("1").count(), 1.0);
    EXPECT_DOUBLE_EQ(parse_duration("  1.5  ").count(), 1.5);
    EXPECT_DOUBLE_EQ(parse_duration("-2").count(), -2.0);
    EXPECT_DOUBLE_EQ(parse_duration("1 s").count(), 1.0);
    EXPECT_DOUBLE_EQ(parse_duration("1s").count(), 1.0);
    EXPECT_DOUBLE_EQ(parse_duration("-1 S").count(), -1.0);
    EXPECT_DOUBLE_EQ(parse_duration("1 ms").count(), 1e-3);
    EXPECT_DOUBLE_EQ(parse_duration("1ms").count(), 1e-3);
    EXPECT_DOUBLE_EQ(parse_duration("2.5 ms").count(), 2.5e-3);
    EXPECT_DOUBLE_EQ(parse_duration("1 us").count(), 1e-6);
    EXPECT_DOUBLE_EQ(parse_duration("1 µs").count(), 1e-6);
    EXPECT_DOUBLE_EQ(parse_duration("1 us").count(), 1e-6);
    EXPECT_DOUBLE_EQ(parse_duration("3 US").count(), 3e-6);
    EXPECT_DOUBLE_EQ(parse_duration("1 ns").count(), 1e-9);
    EXPECT_DOUBLE_EQ(parse_duration("10 NS").count(), 10e-9);
    EXPECT_DOUBLE_EQ(parse_duration("1 m").count(), 60.0);
    EXPECT_DOUBLE_EQ(parse_duration("2 M").count(), 120.0);
    EXPECT_DOUBLE_EQ(parse_duration("0.5 m").count(), 30.0);
    EXPECT_DOUBLE_EQ(parse_duration("-1 m").count(), -60.0);
    EXPECT_DOUBLE_EQ(parse_duration("1 min").count(), 60.0);
    EXPECT_DOUBLE_EQ(parse_duration("2 MIN").count(), 120.0);
    EXPECT_DOUBLE_EQ(parse_duration("1 h").count(), 3600.0);
    EXPECT_DOUBLE_EQ(parse_duration("0.5 H").count(), 1800.0);
    EXPECT_DOUBLE_EQ(parse_duration("1 d").count(), 86400.0);
    EXPECT_DOUBLE_EQ(parse_duration("-2 D").count(), -172800.0);
    EXPECT_DOUBLE_EQ(parse_duration("1e3 s").count(), 1000.0);
    EXPECT_DOUBLE_EQ(parse_duration("1e-3 s").count(), 1e-3);
    EXPECT_DOUBLE_EQ(parse_duration("2.5E-3 min").count(), 0.15);

    // Invalid inputs
    EXPECT_THROW(parse_duration(""), std::invalid_argument);
    EXPECT_THROW(parse_duration("   "), std::invalid_argument);
    EXPECT_THROW(parse_duration("abc"), std::invalid_argument);
    EXPECT_THROW(parse_duration("1.2.3 s"), std::invalid_argument);
    EXPECT_THROW(parse_duration("1 s extra"), std::invalid_argument);
    EXPECT_THROW(parse_duration("--1 s"), std::invalid_argument);
    EXPECT_THROW(parse_duration("1..0 s"), std::invalid_argument);
    EXPECT_THROW(parse_duration("e3 s"), std::invalid_argument);

    // Unknown units
    EXPECT_THROW(parse_duration("1 sec"), std::invalid_argument);
    EXPECT_THROW(parse_duration("1 mins"), std::invalid_argument);
    EXPECT_THROW(parse_duration("1 hr"), std::invalid_argument);
    EXPECT_THROW(parse_duration("1 month"), std::invalid_argument);
    EXPECT_THROW(parse_duration("1 y"), std::invalid_argument);

    // Ambiguous or malformed unit combinations
    EXPECT_THROW(parse_duration("1m s"), std::invalid_argument);
    EXPECT_THROW(parse_duration("1mm"), std::invalid_argument);
    EXPECT_THROW(parse_duration("1mms"), std::invalid_argument);

    // Range / non-finite
    EXPECT_THROW(parse_duration("1e309 s"), std::out_of_range);
    EXPECT_THROW(parse_duration("nan s"), std::invalid_argument);
    EXPECT_THROW(parse_duration("inf s"), std::invalid_argument);
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

TEST(UtilsTest, ParseOptional) {
    // Pass-through
    EXPECT_EQ(parse_optional(""), std::optional<std::string>{""});
    EXPECT_EQ(parse_optional("foo"), std::optional<std::string>{"foo"});
    EXPECT_EQ(parse_optional("  foo  "), std::optional<std::string>{"  foo  "});
    EXPECT_EQ(parse_optional("0"), std::optional<std::string>{"0"});
    EXPECT_EQ(parse_optional("1"), std::optional<std::string>{"1"});
    EXPECT_EQ(parse_optional("true"), std::optional<std::string>{"true"});

    // Disabled keywords (case-insensitive, ignores surrounding whitespace)
    EXPECT_EQ(parse_optional("false"), std::nullopt);
    EXPECT_EQ(parse_optional(" FALSE "), std::nullopt);
    EXPECT_EQ(parse_optional("no"), std::nullopt);
    EXPECT_EQ(parse_optional("\tNO\n"), std::nullopt);
    EXPECT_EQ(parse_optional("off"), std::nullopt);
    EXPECT_EQ(parse_optional("  oFf  "), std::nullopt);
    EXPECT_EQ(parse_optional("disable"), std::nullopt);
    EXPECT_EQ(parse_optional("DISABLED"), std::nullopt);
    EXPECT_EQ(parse_optional("none"), std::nullopt);
    EXPECT_EQ(parse_optional(" n/a "), std::nullopt);
    EXPECT_EQ(parse_optional("NA"), std::nullopt);

    // Must be a full match (no partial matches)
    EXPECT_EQ(parse_optional("falsehood"), std::optional<std::string>{"falsehood"});
    EXPECT_EQ(parse_optional("disabled_now"), std::optional<std::string>{"disabled_now"});
    EXPECT_EQ(parse_optional("n/a2"), std::optional<std::string>{"n/a2"});
    EXPECT_EQ(parse_optional("naive"), std::optional<std::string>{"naive"});
}

TEST(UtilsTest, ParseMemoryTypeFromStream) {
    {
        std::stringstream ss("DEVICE");
        MemoryType v{};
        ss >> v;
        EXPECT_FALSE(ss.fail());
        EXPECT_EQ(v, MemoryType::DEVICE);
    }
    {
        std::stringstream ss("pinned_host");
        MemoryType v{};
        ss >> v;
        EXPECT_FALSE(ss.fail());
        EXPECT_EQ(v, MemoryType::PINNED_HOST);
    }
    {
        std::stringstream ss("PINNED");
        MemoryType v{};
        ss >> v;
        EXPECT_FALSE(ss.fail());
        EXPECT_EQ(v, MemoryType::PINNED_HOST);
    }
    {
        std::stringstream ss("pinned-host");
        MemoryType v{};
        ss >> v;
        EXPECT_FALSE(ss.fail());
        EXPECT_EQ(v, MemoryType::PINNED_HOST);
    }
    {
        std::stringstream ss(" host ");
        MemoryType v{};
        ss >> v;
        EXPECT_FALSE(ss.fail());
        EXPECT_EQ(v, MemoryType::HOST);
    }
}

TEST(UtilsTest, ParseMemoryTypeRejectsInvalidToken) {
    {
        std::stringstream ss("GPU");
        MemoryType v{};
        ss >> v;
        EXPECT_TRUE(ss.fail());
    }
    {
        std::stringstream ss("");
        MemoryType v{};
        ss >> v;
        EXPECT_TRUE(ss.fail());
    }
}

TEST(UtilsTest, ParseStringMemoryType) {
    EXPECT_EQ(parse_string<MemoryType>("DEVICE"), MemoryType::DEVICE);
    EXPECT_EQ(parse_string<MemoryType>("device"), MemoryType::DEVICE);
    EXPECT_EQ(parse_string<MemoryType>(" PINNED_HOST "), MemoryType::PINNED_HOST);
    EXPECT_EQ(parse_string<MemoryType>("pinned"), MemoryType::PINNED_HOST);
    EXPECT_EQ(parse_string<MemoryType>("pinned-host"), MemoryType::PINNED_HOST);
    EXPECT_EQ(parse_string<MemoryType>("HOST"), MemoryType::HOST);
    EXPECT_THROW(parse_string<MemoryType>("gpu"), std::invalid_argument);
    EXPECT_THROW(parse_string<MemoryType>(""), std::invalid_argument);
    EXPECT_THROW(parse_string<MemoryType>("   "), std::invalid_argument);
}

TEST(UtilsTest, ParseStringList) {
    using ::testing::ElementsAre;

    EXPECT_THAT(parse_string_list("a,b,c"), ElementsAre("a", "b", "c"));
    EXPECT_THAT(parse_string_list("  a  ,  b  ,  c  "), ElementsAre("a", "b", "c"));
    EXPECT_THAT(parse_string_list("single"), ElementsAre("single"));
    EXPECT_THAT(parse_string_list(""), ElementsAre());
    EXPECT_THAT(parse_string_list("a,,c"), ElementsAre("a", "", "c"));
    EXPECT_THAT(parse_string_list(",a,b,"), ElementsAre("", "a", "b", ""));
    EXPECT_THAT(parse_string_list("a:b:c", ':'), ElementsAre("a", "b", "c"));
    EXPECT_THAT(parse_string_list("  a : b : c  ", ':'), ElementsAre("a", "b", "c"));
}

TEST(UtilsTest, EscapeCharsEmpty) {
    EXPECT_EQ(escape_chars(""), "");
    EXPECT_EQ(escape_chars("", "abc"), "");
}

TEST(UtilsTest, EscapeCharsCleanPassthrough) {
    // Characters not in the escape set are copied verbatim.
    EXPECT_EQ(escape_chars("hello world"), "hello world");
    EXPECT_EQ(escape_chars("hello world", "xyz"), "hello world");
}

TEST(UtilsTest, EscapeCharsNamedSequences) {
    // Default set: named control characters map to their two-char sequences.
    EXPECT_EQ(escape_chars("\b"), "\\b");
    EXPECT_EQ(escape_chars("\f"), "\\f");
    EXPECT_EQ(escape_chars("\n"), "\\n");
    EXPECT_EQ(escape_chars("\r"), "\\r");
    EXPECT_EQ(escape_chars("\t"), "\\t");
}

TEST(UtilsTest, EscapeCharsQuoteAndBackslash) {
    EXPECT_EQ(escape_chars("\""), "\\\"");
    EXPECT_EQ(escape_chars("\\"), "\\\\");
    EXPECT_EQ(escape_chars("say \"hi\""), "say \\\"hi\\\"");
    EXPECT_EQ(escape_chars("a\\b"), "a\\\\b");
}

TEST(UtilsTest, EscapeCharsControlCharsUnicodeEscape) {
    // Control characters without a named escape use \u00XX.
    EXPECT_EQ(escape_chars(std::string(1, '\x00')), "\\u0000");
    EXPECT_EQ(escape_chars(std::string(1, '\x01')), "\\u0001");
    EXPECT_EQ(escape_chars(std::string(1, '\x1e')), "\\u001e");
    EXPECT_EQ(escape_chars(std::string(1, '\x1f')), "\\u001f");
    // 0x20 (space) is NOT a control character — passes through.
    EXPECT_EQ(escape_chars(" "), " ");
}

TEST(UtilsTest, EscapeCharsMixed) {
    // Mix of clean text, named escapes, and hex escapes.
    EXPECT_EQ(escape_chars("a\nb"), "a\\nb");
    EXPECT_EQ(escape_chars("ke\"y"), "ke\\\"y");
    EXPECT_EQ(escape_chars("a\x01z"), "a\\u0001z");
}

TEST(UtilsTest, EscapeCharsCustomSet) {
    // Only the characters in the custom set are escaped.
    EXPECT_EQ(escape_chars("a@b", "@"), "a\\@b");
    EXPECT_EQ(escape_chars("a\nb", "@"), "a\nb");  // \n not in set, passes through
    EXPECT_EQ(escape_chars("a@b\nc", "@"), "a\\@b\nc");
}

TEST(UtilsTest, ParseStringListMemoryTypes) {
    using ::testing::ElementsAre;

    {
        std::vector<MemoryType> types;
        for (const auto& token : parse_string_list("DEVICE, PINNED_HOST, HOST")) {
            types.push_back(parse_string<MemoryType>(token));
        }
        EXPECT_THAT(
            types,
            ElementsAre(MemoryType::DEVICE, MemoryType::PINNED_HOST, MemoryType::HOST)
        );
    }

    {
        std::vector<MemoryType> types;
        for (const auto& token : parse_string_list("device, pinned, host")) {
            types.push_back(parse_string<MemoryType>(token));
        }
        EXPECT_THAT(
            types,
            ElementsAre(MemoryType::DEVICE, MemoryType::PINNED_HOST, MemoryType::HOST)
        );
    }

    // List with empty item should fail
    EXPECT_THROW(
        {
            std::vector<MemoryType> types;
            for (const auto& token : parse_string_list("DEVICE, , HOST")) {
                types.push_back(parse_string<MemoryType>(token));
            }
        },
        std::invalid_argument
    );
}
