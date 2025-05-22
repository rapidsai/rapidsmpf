/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstring>

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

TEST(OptionsTest, GetOptionWrongTypeThrows) {
    std::unordered_map<std::string, std::string> strings = {{"myoption", "not an int"}};
    Options opts(strings);
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

TEST(OptionsTest, CaseSensitiveKeys) {
    std::unordered_map<std::string, std::string> strings = {
        {"key", "lower-key"}, {"KEY", "upper-key"}
    };
    EXPECT_THROW(Options opts(strings), std::invalid_argument);
}

TEST(OptionsTest, GetStringsReturnsAllStoredOptions) {
    std::unordered_map<std::string, std::string> strings = {
        {"option1", "value1"}, {"option2", "value2"}, {"Option3", "value3"}
    };

    Options opts(strings);
    auto result = opts.get_strings();

    EXPECT_EQ(result.size(), 3);
    EXPECT_EQ(result["option1"], "value1");
    EXPECT_EQ(result["option2"], "value2");
    EXPECT_EQ(result["option3"], "value3");  // Keys are always lower case.
}

TEST(OptionsTest, GetStringsReturnsEmptyMapIfNoOptions) {
    Options opts;
    auto result = opts.get_strings();

    EXPECT_TRUE(result.empty());
}

TEST(OptionsTest, SerializeDeserializeRoundTripPreservesData) {
    std::unordered_map<std::string, std::string> strings = {
        {"alpha", "1"}, {"beta", "two"}, {"gamma", "3.14"}, {"empty", ""}
    };

    Options original(strings);
    auto buffer = original.serialize();
    Options deserialized = Options::deserialize(buffer);

    auto roundtrip = deserialized.get_strings();
    EXPECT_EQ(roundtrip.size(), strings.size());
    for (const auto& [key, value] : strings) {
        EXPECT_EQ(roundtrip[key], value);
    }
}

TEST(OptionsTest, SerializeDeserializeEmptyOptions) {
    Options empty;
    auto buffer = empty.serialize();

    // Buffer should contain only the count (0)
    EXPECT_EQ(buffer.size(), sizeof(uint64_t));

    Options deserialized = Options::deserialize(buffer);
    EXPECT_TRUE(deserialized.get_strings().empty());
}

TEST(OptionsTest, DeserializeThrowsOnBufferTooSmallForCount) {
    std::vector<std::byte> buffer(sizeof(std::uint64_t) - 1);
    EXPECT_THROW(static_cast<void>(Options::deserialize(buffer)), std::invalid_argument);
}

TEST(OptionsTest, DeserializeThrowsOnBufferTooSmallForHeader) {
    // Buffer has count = 2 but no offsets/data
    uint64_t count = 2;
    std::vector<std::byte> buffer(sizeof(uint64_t));
    std::memcpy(buffer.data(), &count, sizeof(uint64_t));
    EXPECT_THROW(static_cast<void>(Options::deserialize(buffer)), std::invalid_argument);
}

TEST(OptionsTest, DeserializeThrowsOnOffsetOutOfBounds) {
    std::unordered_map<std::string, std::string> strings = {{"key", "value"}};
    Options opts(strings);
    auto buffer = opts.serialize();

    // Corrupt count to 1 (valid)
    uint64_t count = 1;
    std::memcpy(buffer.data(), &count, sizeof(uint64_t));

    // Corrupt one offset to be out-of-bounds (key offset)
    auto bad_offset = static_cast<uint64_t>(buffer.size() + 100);
    std::memcpy(buffer.data() + sizeof(uint64_t), &bad_offset, sizeof(uint64_t));

    EXPECT_THROW(static_cast<void>(Options::deserialize(buffer)), std::out_of_range);
}

TEST(OptionsTest, SerializeThrowsIfOptionValueIsSet) {
    std::unordered_map<std::string, std::string> strings = {{"level", "5"}};
    Options opts(strings);
    static_cast<void>(opts.get<int>("level", make_factory<int>(0, [](auto s) {
                                        return std::stoi(s);
                                    })));

    // Expect serialize() to throw because the option value has been accessed.
    EXPECT_THROW(static_cast<void>(opts.serialize()), std::invalid_argument);
}
