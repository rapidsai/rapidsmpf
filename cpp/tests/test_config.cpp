/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rapidsmpf/config.hpp>

using namespace rapidsmpf::config;

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

template <typename T>
OptionFactory<T> make_factory(T default_value, std::function<T(std::string)> parser) {
    return [=](std::string const& s) -> T {
        if (s.empty()) {
            return default_value;
        }
        return parser(s);
    };
}

TEST(OptionsTest, GetOptionCorrectTypeSetExplicitly) {
    std::unordered_map<std::string, OptionValue> options = {{"myoption", OptionValue(42)}
    };
    Options opts(options);
    auto value = opts.get<int>("myoption", make_factory<int>(0, [](auto s) {
                                   return std::stoi(s);
                               }));
    EXPECT_EQ(value, 42);
}

TEST(OptionsTest, GetOptionWrongTypeThrows) {
    std::unordered_map<std::string, OptionValue> options = {
        {"myoption", OptionValue("not an int")}
    };
    Options opts(options);
    EXPECT_THROW(
        {
            opts.get<int>("myoption", make_factory<int>(0, [](auto s) {
                              return std::stoi(s);
                          }));
        },
        std::invalid_argument
    );
}

TEST(OptionsTest, GetUnsetOptionUsesFactoryWithDefaultValue) {
    Options opts;
    auto value = opts.get<std::string>(
        "newoption", make_factory<std::string>("default", [](auto s) { return s; })
    );
    EXPECT_EQ(value, "default");
}

TEST(OptionsTest, GetUnsetOptionUsesFactoryWithStringValue) {
    std::unordered_map<std::string, std::string> strings = {{"level", "5"}};
    Options opts(strings);
    auto value =
        opts.get<int>("level", make_factory<int>(0, [](auto s) { return std::stoi(s); }));
    EXPECT_EQ(value, 5);
}
