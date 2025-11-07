/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <cstring>
#include <mutex>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <unistd.h>

#include <rapidsmpf/config.hpp>
#include <rapidsmpf/utils.hpp>


extern char** environ;

namespace rapidsmpf::config {

Options::Options(std::unordered_map<std::string, OptionValue> options)
    : shared_{std::make_shared<detail::SharedOptions>()} {
    // insert, trim and lower case all keys.
    auto& opts = shared_->options;
    opts.reserve(options.size());
    for (auto&& [key, value] : options) {
        auto new_key = rapidsmpf::to_lower(rapidsmpf::trim(key));
        RAPIDSMPF_EXPECTS(
            opts.emplace(std::move(new_key), std::move(value)).second,
            "option keys must be case-insensitive",
            std::invalid_argument
        );
    }
}

namespace {
// Helper function to get OptionValue map from options-as-strings map.
std::unordered_map<std::string, OptionValue> from_options_as_strings(
    std::unordered_map<std::string, std::string>&& options_as_strings
) {
    std::unordered_map<std::string, OptionValue> ret;
    for (auto&& [key, val] : options_as_strings) {
        ret.emplace(std::move(key), OptionValue(std::move(val)));
    }
    return ret;
}
}  // namespace

Options::Options(std::unordered_map<std::string, std::string> options_as_strings)
    : Options(from_options_as_strings(std::move(options_as_strings))) {};

bool Options::insert_if_absent(std::string const& key, std::string option_as_string) {
    return insert_if_absent({{key, option_as_string}});
}

std::size_t Options::insert_if_absent(
    std::unordered_map<std::string, std::string> options_as_strings
) {
    auto& shared = *shared_;
    std::lock_guard<std::mutex> lock(shared.mutex);
    std::size_t ret = 0;
    for (auto&& [key, val] : options_as_strings) {
        auto new_key = rapidsmpf::to_lower(rapidsmpf::trim(key));
        if (shared.options.insert({std::move(new_key), OptionValue(std::move(val))})
                .second)
        {
            ++ret;
        }
    }
    return ret;
}

std::unordered_map<std::string, std::string> Options::get_strings() const {
    auto const& shared = *shared_;
    std::unordered_map<std::string, std::string> ret;
    std::lock_guard<std::mutex> lock(shared.mutex);
    for (const auto& [key, option] : shared.options) {
        ret[key] = option.get_value_as_string();
    }
    return ret;
}

std::vector<std::uint8_t> Options::serialize() const {
    auto const& shared = *shared_;
    std::lock_guard<std::mutex> lock(shared.mutex);

    std::size_t const count = shared.options.size();
    std::size_t const header_size = (1 + 2 * count) * sizeof(uint64_t);

    std::size_t data_size = 0;
    for (auto const& [key, option] : shared.options) {
        data_size += key.size() + option.get_value_as_string().size();
    }

    std::vector<std::uint8_t> buffer(header_size + data_size);
    std::uint8_t* base = buffer.data();

    // Write count (number of key-value pairs).
    {
        auto const count_ = static_cast<uint64_t>(count);
        std::memcpy(base, &count_, sizeof(uint64_t));
    }

    // Write offsets and data.
    std::size_t offset_index = 1;  // Offsets starts after `count`.
    std::size_t data_offset = header_size;
    for (auto const& [key, option] : shared.options) {
        RAPIDSMPF_EXPECTS(
            !option.get_value().has_value(),
            "cannot serialize already parsed (accessed) option values",
            std::invalid_argument
        );
        std::string const& value = option.get_value_as_string();

        auto key_offset = static_cast<uint64_t>(data_offset);
        auto value_offset = static_cast<uint64_t>(key_offset + key.size());

        // Write offsets
        std::memcpy(
            base + offset_index * sizeof(uint64_t), &key_offset, sizeof(uint64_t)
        );
        std::memcpy(
            base + (offset_index + 1) * sizeof(uint64_t), &value_offset, sizeof(uint64_t)
        );
        offset_index += 2;

        // Write data
        std::memcpy(base + key_offset, key.data(), key.size());
        std::memcpy(base + value_offset, value.data(), value.size());

        data_offset = static_cast<std::size_t>(value_offset + value.size());
    }

    return buffer;
}

Options Options::deserialize(std::vector<std::uint8_t> const& buffer) {
    const std::uint8_t* base = buffer.data();
    std::size_t total_size = buffer.size();

    // Read number of key-value pairs
    uint64_t count = 0;
    RAPIDSMPF_EXPECTS(
        total_size >= sizeof(uint64_t),
        "buffer is too small to contain count",
        std::invalid_argument
    );
    std::memcpy(&count, base, sizeof(uint64_t));
    std::size_t const header_size = (1 + 2 * count) * sizeof(uint64_t);
    RAPIDSMPF_EXPECTS(
        header_size <= total_size,
        "buffer is too small for header with declared count",
        std::invalid_argument
    );

    // Read offsets
    std::vector<uint64_t> key_offsets(count);
    std::vector<uint64_t> value_offsets(count);
    for (uint64_t i = 0; i < count; ++i) {
        std::memcpy(
            &key_offsets[i], base + (1 + 2 * i) * sizeof(uint64_t), sizeof(uint64_t)
        );
        std::memcpy(
            &value_offsets[i], base + (1 + 2 * i + 1) * sizeof(uint64_t), sizeof(uint64_t)
        );
    }

    // Reconstruct the key-value pairs
    std::unordered_map<std::string, std::string> ret;
    for (uint64_t i = 0; i < count; ++i) {
        uint64_t const key_offset = key_offsets[i];
        uint64_t const value_offset = value_offsets[i];

        RAPIDSMPF_EXPECTS(
            key_offset < total_size && value_offset < total_size
                && key_offset < value_offset,
            "invalid offsets in serialized buffer",
            std::out_of_range
        );

        std::size_t const key_len = value_offset - key_offset;
        std::size_t const value_len =
            (i + 1 < count ? key_offsets[i + 1] : total_size) - value_offset;

        if (key_offset + key_len > total_size || value_offset + value_len > total_size) {
            throw std::out_of_range("Deserialization offset exceeds buffer size");
        }
        std::string key(reinterpret_cast<const char*>(base + key_offset), key_len);
        std::string val(reinterpret_cast<const char*>(base + value_offset), value_len);
        ret.emplace(std::move(key), std::move(val));
    }

    return Options(ret);
}

void get_environment_variables(
    std::unordered_map<std::string, std::string>& output, std::string const& key_regex
) {
    RAPIDSMPF_EXPECTS(
        std::regex(key_regex).mark_count() == 1,
        "key_regex must contain exactly one capture group (e.g., \"RAPIDSMPF_(.*)\")",
        std::invalid_argument
    );

    std::regex pattern(key_regex + "=(.*)");
    for (char** env = environ; *env != nullptr; ++env) {
        std::string entry(*env);
        std::smatch match;
        if (std::regex_match(entry, match, pattern)) {
            if (match.size() == 3) {  // match[1]: captured key, match[2]: value
                output.insert({rapidsmpf::to_lower(match[1].str()), match[2].str()});
            }
        }
    }
}

std::unordered_map<std::string, std::string> get_environment_variables(
    std::string const& key_regex
) {
    std::unordered_map<std::string, std::string> ret;
    get_environment_variables(ret, key_regex);
    return ret;
}

}  // namespace rapidsmpf::config
