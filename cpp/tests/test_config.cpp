/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstring>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cudf/utilities/memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>

#include <rapidsmpf/config.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/pinned_memory_resource.hpp>
#include <rapidsmpf/rmm_resource_adaptor.hpp>
#include <rapidsmpf/statistics.hpp>
#include <rapidsmpf/utils/misc.hpp>

#include "utils.hpp"

using namespace rapidsmpf::config;
using namespace rapidsmpf;

TEST(OptionsTest, EnvReturnsMatchingVariables) {
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

TEST(OptionsTest, EnvOutputMapIsPopulated) {
    setenv("RAPIDSMPF_ANOTHER_VAR", "another_value", 1);

    std::unordered_map<std::string, std::string> output;
    get_environment_variables(output, "RAPIDSMPF_(.*)");

    ASSERT_TRUE(output.find("ANOTHER_VAR") != output.end());
    EXPECT_EQ(output["ANOTHER_VAR"], "another_value");
}

TEST(OptionsTest, EnvDoesNotOverwriteExistingKey) {
    setenv("RAPIDSMPF_EXISTING_VAR", "env_value", 1);

    std::unordered_map<std::string, std::string> output;
    output["EXISTING_VAR"] = "original_value";

    get_environment_variables(output, "RAPIDSMPF_(.*)");

    // The value should remain as the original, not overwritten by the environment
    EXPECT_EQ(output["EXISTING_VAR"], "original_value");
}

TEST(OptionsTest, EnvThrowsIfNoCaptureGroup) {
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

TEST(OptionsTest, InsertIfAbsentInsertsNewKey) {
    Options opts;
    bool inserted = opts.insert_if_absent("somekey", "42");

    EXPECT_TRUE(inserted);
    int value = opts.get<int>("somekey", make_factory<int>(0, [](auto s) {
                                  return std::stoi(s);
                              }));
    EXPECT_EQ(value, 42);
}

TEST(OptionsTest, InsertIfAbsentDoesNotOverwriteExistingKey) {
    std::unordered_map<std::string, std::string> strings = {{"somekey", "123"}};
    Options opts(strings);

    // This should not overwrite the existing value
    bool inserted = opts.insert_if_absent("SomeKey", "999");
    EXPECT_FALSE(inserted);

    int value = opts.get<int>("somekey", make_factory<int>(0, [](auto s) {
                                  return std::stoi(s);
                              }));
    EXPECT_EQ(value, 123);
}

TEST(OptionsTest, InsertIfAbsentMapInsertsNewKeysOnly) {
    Options opts;
    // Add existing key
    opts.insert_if_absent("existingkey", "111");
    // Prepare map with existing and new keys
    std::unordered_map<std::string, std::string> new_options = {
        {"existingkey", "222"},  // Should NOT be inserted
        {"newkey1", "333"},  // Should be inserted
        {"newkey2", "444"}  // Should be inserted
    };
    // Insert using map overload
    std::size_t inserted_count = opts.insert_if_absent(std::move(new_options));

    // Check count: only newkey1 and newkey2 should be inserted
    EXPECT_EQ(inserted_count, 2);

    // Check existing key remains unchanged
    int value = opts.get<int>("existingkey", make_factory<int>(0, [](auto s) {
                                  return std::stoi(s);
                              }));
    EXPECT_EQ(value, 111);

    // Check new keys are inserted
    value = opts.get<int>("newkey1", make_factory<int>(0, [](auto s) {
                              return std::stoi(s);
                          }));
    EXPECT_EQ(value, 333);

    value = opts.get<int>("newkey2", make_factory<int>(0, [](auto s) {
                              return std::stoi(s);
                          }));
    EXPECT_EQ(value, 444);
}

TEST(OptionsTest, InsertIfAbsentTypedInsertsAndGetReturnsValue) {
    Options opts;

    bool inserted = opts.insert_if_absent<int>("  SomeKey  ", 42);
    EXPECT_TRUE(inserted);

    // Factory should not be used because the value is already set.
    auto value = opts.get<int>("somekey", make_factory<int>(0, [](auto) {
                                   ADD_FAILURE() << "factory should not be called";
                                   return 0;
                               }));
    EXPECT_EQ(value, 42);

    // Typed values have no string representation.
    auto strings = opts.get_strings();
    EXPECT_TRUE(strings.at("somekey").empty());

    // Typed insertion makes the instance unserializable.
    EXPECT_THROW(static_cast<void>(opts.serialize()), std::invalid_argument);
}

TEST(OptionsTest, InsertIfAbsentTypedDoesNotOverwriteExistingKey) {
    Options opts;
    EXPECT_TRUE(opts.insert_if_absent<int>("k", 1));

    // Should not overwrite.
    EXPECT_FALSE(opts.insert_if_absent<int>("K", 2));

    auto value = opts.get<int>("k", make_factory<int>(0, [](auto) {
                                   ADD_FAILURE() << "factory should not be called";
                                   return 0;
                               }));
    EXPECT_EQ(value, 1);
}

TEST(OptionsTest, InsertIfAbsentTypedThrowsOnTypeMismatch) {
    Options opts;

    // Insert as int
    EXPECT_TRUE(opts.insert_if_absent<int>("value", 42));

    // Attempting to retrieve as a different type should throw std::bad_any_cast
    EXPECT_THROW(
        std::ignore = opts.get<double>(
            "value",
            make_factory<double>(
                0,
                [](auto) {
                    ADD_FAILURE() << "factory should not be called";
                    return 0.0;
                }
            )
        );
        , std::invalid_argument
    );
}

TEST(OptionsTest, InsertIfAbsentStringViewUsesStringOverloadAndRemainsSerializable) {
    Options opts;

    std::string_view sv = "5";
    bool inserted = opts.insert_if_absent("level", sv);
    EXPECT_TRUE(inserted);

    // Should store the string representation.
    auto strings = opts.get_strings();
    ASSERT_TRUE(strings.find("level") != strings.end());
    EXPECT_EQ(strings["level"], "5");

    // Should remain serializable, since no typed value was inserted and we have not
    // accessed options via get().
    EXPECT_NO_THROW(static_cast<void>(opts.serialize()));
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

TEST(OptionValueTest, TypedCtorStoresValue) {
    OptionValue ov(123);

    EXPECT_TRUE(ov.get_value().has_value());
    EXPECT_TRUE(ov.get_value_as_string().empty());
    EXPECT_EQ(std::any_cast<int>(ov.get_value()), 123);
}

TEST(OptionValueTest, TypedCtorMovesValue) {
    std::vector<int> v{1, 2, 3};
    auto* data_before = v.data();

    OptionValue ov(std::move(v));
    EXPECT_TRUE(ov.get_value().has_value());
    EXPECT_TRUE(ov.get_value_as_string().empty());

    auto const& stored = std::any_cast<std::vector<int> const&>(ov.get_value());
    EXPECT_EQ(stored, (std::vector<int>{1, 2, 3}));
    EXPECT_TRUE(v.empty() || v.data() != data_before);
}

TEST(OptionValueTest, TypedCtorDoesNotAllowSetValueAgain) {
    OptionValue ov(1);
    EXPECT_THROW(ov.set_value(std::make_any<int>(2)), std::invalid_argument);
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

TEST(OptionsTest, DeserializeThrowsOnCrcMismatch) {
    std::unordered_map<std::string, std::string> strings = {{"alpha", "1"}};
    Options opts(strings);
    auto buffer = opts.serialize();
    ASSERT_GE(buffer.size(), static_cast<size_t>(8 + sizeof(uint64_t) + 4));
    // Flip one byte in the data region (just before CRC trailer)
    if (buffer.size() > (8 + sizeof(uint64_t) + 4)) {
        buffer[buffer.size() - 5] ^= 0xFF;
        EXPECT_THROW(
            static_cast<void>(Options::deserialize(buffer)), std::invalid_argument
        );
    }
}

TEST(OptionsTest, SerializeDeserializeEmptyOptions) {
    Options empty;
    auto buffer = empty.serialize();

    // Buffer contains MAGIC+ver (8), count (8) and CRC32 (4)
    EXPECT_EQ(buffer.size(), 8 + sizeof(uint64_t) + 4);

    Options deserialized = Options::deserialize(buffer);
    EXPECT_TRUE(deserialized.get_strings().empty());
}

TEST(OptionsTest, DeserializeThrowsOnBufferTooSmallForCount) {
    std::vector<std::uint8_t> buffer(sizeof(std::uint64_t) - 1);
    EXPECT_THROW(static_cast<void>(Options::deserialize(buffer)), std::invalid_argument);
}

TEST(OptionsTest, DeserializeThrowsOnBufferTooSmallForHeader) {
    // Buffer has MAGIC+ver/flags and count = 2 but no offsets/data
    uint64_t count = 2;
    std::vector<std::uint8_t> buffer(8 + sizeof(uint64_t));
    buffer[0] = 'R';
    buffer[1] = 'M';
    buffer[2] = 'P';
    buffer[3] = 'F';
    buffer[4] = 1;  // version
    buffer[5] = 0x01;  // flags: CRC present (won't be validated due to too-small buffer)
    buffer[6] = 0;
    buffer[7] = 0;  // reserved
    std::memcpy(buffer.data() + 8, &count, sizeof(uint64_t));
    EXPECT_THROW(static_cast<void>(Options::deserialize(buffer)), std::invalid_argument);
}

TEST(OptionsTest, DeserializeThrowsOnOffsetOutOfBounds) {
    std::unordered_map<std::string, std::string> strings = {{"key", "value"}};
    Options opts(strings);
    auto buffer = opts.serialize();

    // Corrupt count to 1 (valid)
    uint64_t count = 1;
    // Count is stored after 8-byte prelude
    std::memcpy(buffer.data() + 8, &count, sizeof(uint64_t));

    // Corrupt one offset to be out-of-bounds (key offset)
    auto bad_offset = static_cast<uint64_t>(buffer.size() + 100);
    // First key offset follows prelude (8) + count (8)
    std::memcpy(buffer.data() + 8 + sizeof(uint64_t), &bad_offset, sizeof(uint64_t));

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

TEST(SerializationLimits, ExceedMaxKeyLength) {
    std::string long_key(4097, 'k');  // 4097 bytes, exceeds 4096 limit
    std::unordered_map<std::string, std::string> strings = {{long_key, "value"}};
    Options opts(strings);

    EXPECT_THROW(static_cast<void>(opts.serialize()), std::invalid_argument);
}

TEST(SerializationLimits, ExceedMaxValueLength) {
    std::string long_value(1024 * 1024 + 1, 'v');  // 1048577 bytes, exceeds 1 MiB limit
    std::unordered_map<std::string, std::string> strings = {{"key", long_value}};
    Options opts(strings);

    EXPECT_THROW(static_cast<void>(opts.serialize()), std::invalid_argument);
}

TEST(SerializationLimits, ExceedMaxOptions) {
    Options opts;
    std::unordered_map<std::string, std::string> many_options;

    // Create 65537 options, exceeds 65536 limit
    for (int i = 0; i < 65537; ++i) {
        many_options["key_" + std::to_string(i)] = "value_" + std::to_string(i);
    }
    opts.insert_if_absent(std::move(many_options));

    EXPECT_THROW(static_cast<void>(opts.serialize()), std::invalid_argument);
}

TEST(SerializationLimits, ExceedMaxTotalSize) {
    Options opts;

    // Each value is 1 MiB, so 65 of them would be 65 MiB + overhead, exceeding 64 MiB
    // limit
    std::string large_value(1024 * 1024, 'x');
    std::unordered_map<std::string, std::string> many_options;

    for (int i = 0; i < 65; ++i) {
        many_options["key_" + std::to_string(i)] = large_value;
    }
    opts.insert_if_absent(std::move(many_options));

    EXPECT_THROW(static_cast<void>(opts.serialize()), std::invalid_argument);
}

TEST(OptionsTest, StatisticsFromOptionsEnabledWhenSetToTrue) {
    std::unordered_map<std::string, std::string> strings = {{"statistics", "True"}};
    Options opts(strings);

    rmm::mr::cuda_memory_resource cuda_mr;
    RmmResourceAdaptor mr{&cuda_mr};
    auto stats = Statistics::from_options(&mr, opts);

    ASSERT_NE(stats, nullptr);
    EXPECT_TRUE(stats->enabled());
}

TEST(OptionsTest, StatisticsFromOptionsEnabledWhenSetToOne) {
    std::unordered_map<std::string, std::string> strings = {{"statistics", "1"}};
    Options opts(strings);

    rmm::mr::cuda_memory_resource cuda_mr;
    RmmResourceAdaptor mr{&cuda_mr};
    auto stats = Statistics::from_options(&mr, opts);

    ASSERT_NE(stats, nullptr);
    EXPECT_TRUE(stats->enabled());
}

TEST(OptionsTest, StatisticsFromOptionsDisabledWhenSetToFalse) {
    std::unordered_map<std::string, std::string> strings = {{"statistics", "False"}};
    Options opts(strings);

    rmm::mr::cuda_memory_resource cuda_mr;
    RmmResourceAdaptor mr{&cuda_mr};
    auto stats = Statistics::from_options(&mr, opts);

    ASSERT_NE(stats, nullptr);
    EXPECT_FALSE(stats->enabled());
}

TEST(OptionsTest, StatisticsFromOptionsDisabledByDefault) {
    Options opts;  // Empty options

    rmm::mr::cuda_memory_resource cuda_mr;
    RmmResourceAdaptor mr{&cuda_mr};
    auto stats = Statistics::from_options(&mr, opts);
    EXPECT_TRUE(stats == Statistics::disabled());
    EXPECT_FALSE(stats->enabled());
}

TEST(OptionsTest, PinnedMemoryResourceFromOptionsEnabledWhenSetToTrue) {
    std::unordered_map<std::string, std::string> strings = {{"pinned_memory", "True"}};
    Options opts(strings);

    auto pmr = PinnedMemoryResource::from_options(opts);

    // Should be enabled if system supports it, or Disabled (nullptr) if not
    if (is_pinned_memory_resources_supported()) {
        EXPECT_NE(pmr, PinnedMemoryResource::Disabled);
        EXPECT_NE(pmr, nullptr);
    } else {
        EXPECT_EQ(pmr, PinnedMemoryResource::Disabled);
        EXPECT_EQ(pmr, nullptr);
    }
}

TEST(OptionsTest, PinnedMemoryResourceFromOptionsDisabledWhenSetToFalse) {
    std::unordered_map<std::string, std::string> strings = {{"pinned_memory", "False"}};
    Options opts(strings);

    auto pmr = PinnedMemoryResource::from_options(opts);

    EXPECT_EQ(pmr, PinnedMemoryResource::Disabled);
    EXPECT_EQ(pmr, nullptr);
}

TEST(OptionsTest, PinnedMemoryResourceFromOptionsDisabledByDefault) {
    Options opts;  // Empty options

    auto pmr = PinnedMemoryResource::from_options(opts);

    EXPECT_EQ(pmr, PinnedMemoryResource::Disabled);
    EXPECT_EQ(pmr, nullptr);
}

TEST(OptionsTest, MemoryAvailableFromOptionsCreatesMapWithDeviceLimit) {
    std::unordered_map<std::string, std::string> strings = {
        {"spill_device_limit", "1GiB"}
    };
    Options opts(strings);

    rmm::mr::cuda_memory_resource cuda_mr;
    RmmResourceAdaptor mr{&cuda_mr};
    auto mem_available = memory_available_from_options(&mr, opts);

    // Should contain a DEVICE entry
    ASSERT_TRUE(mem_available.find(MemoryType::DEVICE) != mem_available.end());

    // Should return the configured limit (1 GiB)
    auto available = mem_available[MemoryType::DEVICE]();
    EXPECT_EQ(available, 1_GiB);
}

TEST(OptionsTest, MemoryAvailableFromOptionsUsesPercentageOfTotalMemory) {
    std::unordered_map<std::string, std::string> strings = {
        {"spill_device_limit", "50%"}
    };
    Options opts(strings);

    rmm::mr::cuda_memory_resource cuda_mr;
    RmmResourceAdaptor mr{&cuda_mr};
    auto mem_available = memory_available_from_options(&mr, opts);

    ASSERT_TRUE(mem_available.find(MemoryType::DEVICE) != mem_available.end());

    // Should return 50% of total device memory
    auto [_, total_mem] = rmm::available_device_memory();
    auto expected = rmm::align_down(total_mem / 2, rmm::CUDA_ALLOCATION_ALIGNMENT);
    auto available = mem_available[MemoryType::DEVICE]();
    EXPECT_EQ(available, expected);
}

TEST(OptionsTest, MemoryAvailableFromOptionsUsesDefaultWhenNotSet) {
    Options opts;  // Empty options

    rmm::mr::cuda_memory_resource cuda_mr;
    RmmResourceAdaptor mr{&cuda_mr};
    auto mem_available = memory_available_from_options(&mr, opts);

    ASSERT_TRUE(mem_available.find(MemoryType::DEVICE) != mem_available.end());

    // Should use default of 80%
    auto [_, total_mem] = rmm::available_device_memory();
    auto expected = rmm::align_down(total_mem * 4 / 5, rmm::CUDA_ALLOCATION_ALIGNMENT);
    auto available = mem_available[MemoryType::DEVICE]();
    EXPECT_EQ(available, expected);
}

TEST(OptionsTest, PeriodicSpillCheckFromOptionsParsesMilliseconds) {
    std::unordered_map<std::string, std::string> strings = {
        {"periodic_spill_check", "5ms"}
    };
    Options opts(strings);

    auto duration = periodic_spill_check_from_options(opts);

    ASSERT_TRUE(duration.has_value());
    EXPECT_EQ(duration.value().count(), 0.005);  // 5ms = 0.005s
}

TEST(OptionsTest, PeriodicSpillCheckFromOptionsParsesSeconds) {
    std::unordered_map<std::string, std::string> strings = {
        {"periodic_spill_check", "2"}
    };
    Options opts(strings);

    auto duration = periodic_spill_check_from_options(opts);

    ASSERT_TRUE(duration.has_value());
    EXPECT_EQ(duration.value().count(), 2.0);
}

TEST(OptionsTest, PeriodicSpillCheckFromOptionsDisabledWhenSetToDisabled) {
    std::unordered_map<std::string, std::string> strings = {
        {"periodic_spill_check", "disabled"}
    };
    Options opts(strings);

    auto duration = periodic_spill_check_from_options(opts);

    EXPECT_FALSE(duration.has_value());
}

TEST(OptionsTest, PeriodicSpillCheckFromOptionsUsesDefaultWhenNotSet) {
    Options opts;  // Empty options

    auto duration = periodic_spill_check_from_options(opts);

    ASSERT_TRUE(duration.has_value());
    EXPECT_EQ(duration.value().count(), 0.001);  // Default: 1ms
}

TEST(OptionsTest, StreamPoolFromOptionsCreatesPoolWithSpecifiedSize) {
    std::unordered_map<std::string, std::string> strings = {{"num_streams", "32"}};
    Options opts(strings);

    auto pool = stream_pool_from_options(opts);

    ASSERT_NE(pool, nullptr);
    EXPECT_EQ(pool->get_pool_size(), 32);
}

TEST(OptionsTest, StreamPoolFromOptionsUsesDefaultWhenNotSet) {
    Options opts;  // Empty options

    auto pool = stream_pool_from_options(opts);

    ASSERT_NE(pool, nullptr);
    EXPECT_EQ(pool->get_pool_size(), 16);  // Default: 16
}

TEST(OptionsTest, StreamPoolFromOptionsThrowsOnZeroStreams) {
    std::unordered_map<std::string, std::string> strings = {{"num_streams", "0"}};
    Options opts(strings);

    EXPECT_THROW(stream_pool_from_options(opts), std::invalid_argument);
}

TEST(OptionsTest, BufferResourceFromOptionsCreatesInstanceWithExplicitOptions) {
    std::unordered_map<std::string, std::string> strings = {
        {"statistics", "True"},
        {"pinned_memory", "False"},
        {"spill_device_limit", "1GiB"},
        {"periodic_spill_check", "5ms"},
        {"num_streams", "8"}
    };
    config::Options opts(strings);

    rmm::mr::cuda_memory_resource cuda_mr;
    RmmResourceAdaptor mr{&cuda_mr};
    auto br = BufferResource::from_options(&mr, opts);

    EXPECT_TRUE(br->statistics()->enabled());
    EXPECT_EQ(br->stream_pool().get_pool_size(), 8);
    auto mem_avail = br->memory_available(MemoryType::DEVICE);
    EXPECT_EQ(mem_avail(), 1_GiB);
}

TEST(OptionsTest, BufferResourceFromOptionsUsesDefaultWhenOptionsEmpty) {
    config::Options opts;  // Empty options

    rmm::mr::cuda_memory_resource cuda_mr;
    RmmResourceAdaptor mr{&cuda_mr};
    auto br = BufferResource::from_options(&mr, opts);
    EXPECT_FALSE(br->statistics()->enabled());
    EXPECT_EQ(br->stream_pool().get_pool_size(), 16);
    auto [_, total_mem] = rmm::available_device_memory();
    auto expected = rmm::align_down(total_mem * 4 / 5, rmm::CUDA_ALLOCATION_ALIGNMENT);
    auto mem_avail = br->memory_available(MemoryType::DEVICE);
    EXPECT_EQ(mem_avail(), expected);
}

TEST(OptionsTest, BufferResourceFromOptionsEnablesStatisticsWhenRequested) {
    std::unordered_map<std::string, std::string> strings = {{"statistics", "1"}};
    config::Options opts(strings);

    rmm::mr::cuda_memory_resource cuda_mr;
    RmmResourceAdaptor mr{&cuda_mr};
    auto br = BufferResource::from_options(&mr, opts);

    EXPECT_TRUE(br->statistics()->enabled());
}

TEST(OptionsTest, BufferResourceFromOptionsAcceptsPercentageForDeviceLimit) {
    std::unordered_map<std::string, std::string> strings = {
        {"spill_device_limit", "50%"}
    };
    config::Options opts(strings);

    rmm::mr::cuda_memory_resource cuda_mr;
    RmmResourceAdaptor mr{&cuda_mr};
    auto br = BufferResource::from_options(&mr, opts);

    // Verify device memory limit is 50% of total
    auto [_, total_mem] = rmm::available_device_memory();
    auto expected = rmm::align_down(total_mem / 2, rmm::CUDA_ALLOCATION_ALIGNMENT);
    auto mem_avail = br->memory_available(MemoryType::DEVICE);
    EXPECT_EQ(mem_avail(), expected);
}

TEST(OptionsTest, BufferResourceFromOptionsEnablesPinnedMemoryWhenSupported) {
    if (!is_pinned_memory_resources_supported()) {
        GTEST_SKIP() << "Pinned memory not supported on this system";
    }

    std::unordered_map<std::string, std::string> strings = {{"pinned_memory", "True"}};
    config::Options opts(strings);

    rmm::mr::cuda_memory_resource cuda_mr;
    RmmResourceAdaptor mr{&cuda_mr};
    auto br = BufferResource::from_options(&mr, opts);

    // Should not throw when accessing pinned_mr
    EXPECT_NO_THROW(std::ignore = br->pinned_mr());
}
