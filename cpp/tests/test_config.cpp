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

    Options opts({}, std::move(options));
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

    Options opts(std::move(options_as_strings));
    auto retrieved_option = opts.get<MockOption>("key");

    ASSERT_NE(retrieved_option, nullptr);
    EXPECT_EQ(retrieved_option->value, "fallback-value");
}

TEST(Options, InvalidTypeAccess) {
    std::unordered_map<std::string, std::unique_ptr<Option>> options;
    options["key"] = std::make_unique<MockOption>("value");

    Options opts({}, std::move(options));

    EXPECT_THROW(opts.get<MockOptionNotUsed>("key"), std::invalid_argument);
}

TEST(Options, MissingKeyThrowsNoException) {
    Options opts;
    auto retrieved_option = opts.get<MockOption>("missing-key");

    ASSERT_NE(retrieved_option, nullptr);
    EXPECT_EQ(retrieved_option->value, "");
}

TEST(Options, KeysAreConvertedToLowercase) {
    std::unordered_map<std::string, std::unique_ptr<Option>> options;
    options["KEYUPPER"] = std::make_unique<MockOption>("upper");
    options["KeyMixed"] = std::make_unique<MockOption>("mixed");
    options["keylower"] = std::make_unique<MockOption>("lower");

    Options opts({}, std::move(options));

    // All keys should be accessible in lowercase
    auto opt1 = opts.get<MockOption>("keyupper");
    auto opt2 = opts.get<MockOption>("keymixed");
    auto opt3 = opts.get<MockOption>("keylower");

    ASSERT_NE(opt1, nullptr);
    ASSERT_NE(opt2, nullptr);
    ASSERT_NE(opt3, nullptr);

    EXPECT_EQ(opt1->value, "upper");
    EXPECT_EQ(opt2->value, "mixed");
    EXPECT_EQ(opt3->value, "lower");

    // Original case keys should not be accessible (should fallback to default)
    auto opt4 = opts.get<MockOption>("KEYUPPER");
    auto opt5 = opts.get<MockOption>("KeyMixed");

    ASSERT_NE(opt4, nullptr);
    ASSERT_NE(opt5, nullptr);

    EXPECT_EQ(opt4->value, "");
    EXPECT_EQ(opt5->value, "");
}

TEST(ConfigEnvironmentVariables, ReturnsMatchingVariables) {
    // Set environment variables for testing
    setenv("RAPIDSMPF_TEST_VAR1", "value1", 1);
    setenv("RAPIDSMPF_TEST_VAR2", "value2", 1);
    setenv("OTHER_VAR", "should_not_match", 1);

    auto env_vars = get_environment_variables("RAPIDSMPF_(.*)");

    // Should contain the variables with the prefix stripped
    ASSERT_TRUE(env_vars.find("TEST_VAR1") != env_vars.end());
    ASSERT_TRUE(env_vars.find("TEST_VAR2") != env_vars.end());
    EXPECT_EQ(env_vars["TEST_VAR1"], "value1");
    EXPECT_EQ(env_vars["TEST_VAR2"], "value2");

    // Should not contain non-matching variables
    ASSERT_TRUE(env_vars.find("OTHER_VAR") == env_vars.end());
}

TEST(ConfigEnvironmentVariables, OutputMapIsPopulated) {
    setenv("RAPIDSMPF_ANOTHER_VAR", "another_value", 1);

    std::unordered_map<std::string, std::string> output;
    get_environment_variables(output, "RAPIDSMPF_(.*)");

    ASSERT_TRUE(output.find("ANOTHER_VAR") != output.end());
    EXPECT_EQ(output["ANOTHER_VAR"], "another_value");
}

TEST(ConfigEnvironmentVariables, DoesNotOverwriteExistingKey) {
    setenv("RAPIDSMPF_EXISTING_VAR", "env_value", 1);

    std::unordered_map<std::string, std::string> output;
    output["EXISTING_VAR"] = "original_value";

    get_environment_variables(output, "RAPIDSMPF_(.*)");

    // The value should remain as the original, not overwritten by the environment
    EXPECT_EQ(output["EXISTING_VAR"], "original_value");
}

TEST(ConfigEnvironmentVariables, ThrowsIfNoCaptureGroup) {
    setenv("RAPIDSMPF_NOCAPTURE", "should_fail", 1);

    // Should throw because there is no capture group in the regex
    EXPECT_THROW(get_environment_variables("RAPIDSMPF_.*"), std::invalid_argument);

    std::unordered_map<std::string, std::string> output;
    EXPECT_THROW(
        get_environment_variables(output, "RAPIDSMPF_.*"), std::invalid_argument
    );
}
