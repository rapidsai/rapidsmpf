/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rapidsmpf/config.hpp>

using namespace rapidsmpf::config;

class MockOption : public Option {
  public:
    explicit MockOption(std::string value = "") : value(std::move(value)) {}

    std::string value;
};

class MockOptionNotUsed : public Option {
  public:
    explicit MockOptionNotUsed(std::string value = "") : value(std::move(value)) {}

    std::string value;
};

TEST(Options, RetrieveExistingOption) {
    std::unordered_map<std::string, std::unique_ptr<Option>> options;
    options["key"] = std::make_unique<MockOption>("value");

    Options opts(std::move(options));
    auto retrieved_option = opts.get<MockOption>("key");

    ASSERT_NE(retrieved_option, nullptr);
    EXPECT_EQ(retrieved_option->value, "value");
}

TEST(Options, RetrieveNonExistingOption) {
    Options opts;
    auto retrieved_option = opts.get<MockOption>("nonexistent-key");

    ASSERT_NE(retrieved_option, nullptr);
    EXPECT_EQ(retrieved_option->value, "");
}

TEST(Options, RetrieveOptionWithStringFallback) {
    std::unordered_map<std::string, std::string> options_as_strings;
    options_as_strings["key"] = "fallback-value";

    Options opts({}, std::move(options_as_strings));
    auto retrieved_option = opts.get<MockOption>("key");

    ASSERT_NE(retrieved_option, nullptr);
    EXPECT_EQ(retrieved_option->value, "fallback-value");
}

TEST(Options, InvalidTypeAccess) {
    std::unordered_map<std::string, std::unique_ptr<Option>> options;
    options["key"] = std::make_unique<MockOption>("value");

    Options opts(std::move(options));

    EXPECT_THROW(opts.get<MockOptionNotUsed>("key"), std::invalid_argument);
}

TEST(Options, MissingKeyThrowsNoException) {
    Options opts;
    auto retrieved_option = opts.get<MockOption>("missing-key");

    ASSERT_NE(retrieved_option, nullptr);
    EXPECT_EQ(retrieved_option->value, "");
}
