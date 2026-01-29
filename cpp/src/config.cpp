/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <unistd.h>

#include <rapidsmpf/config.hpp>
#include <rapidsmpf/utils/string.hpp>

namespace {
// Serialization limits and format configuration (implementation details)
constexpr std::size_t MAX_OPTIONS = 64 * (1ull << 10);
constexpr std::size_t MAX_KEY_LEN = 4 * (1ull << 10);
constexpr std::size_t MAX_VALUE_LEN = 1 * (1ull << 20);
constexpr std::size_t MAX_TOTAL_SIZE = 64 * (1ull << 20);

// Format constants
constexpr std::array<std::byte, 4> MAGIC{
    {static_cast<std::byte>('R'),
     static_cast<std::byte>('M'),
     static_cast<std::byte>('P'),
     static_cast<std::byte>('F')}
};
constexpr std::byte FORMAT_VERSION = static_cast<std::byte>(1);
constexpr std::byte FLAG_CRC_PRESENT = static_cast<std::byte>(0x01);
// MAGIC(4) + version(1) + flags(1) + reserved(2)
constexpr std::size_t PRELUDE_SIZE =
    sizeof(MAGIC) + sizeof(FORMAT_VERSION) + sizeof(FLAG_CRC_PRESENT) + 2;
constexpr std::size_t CRC32_SIZE = 4;

std::uint32_t crc32_compute(std::byte const* data, std::size_t length) {
    std::uint32_t crc = 0xFFFFFFFFu;
    for (std::size_t i = 0; i < length; ++i) {
        crc ^= std::to_integer<std::uint32_t>(data[i]);
        for (int bit = 0; bit < 8; ++bit) {
            std::uint32_t mask = -(crc & 1u);
            crc = (crc >> 1) ^ (0xEDB88320u & mask);
        }
    }
    return ~crc;
}

std::size_t validate_crc_and_get_data_limit(
    std::byte const* base,
    std::size_t header_size,
    std::size_t total_size,
    std::byte flags
) {
    std::size_t data_limit = total_size;
    if ((flags & FLAG_CRC_PRESENT) != std::byte(0u)) {
        RAPIDSMPF_EXPECTS(
            total_size >= header_size + CRC32_SIZE,
            "buffer too small for CRC32 trailer",
            std::invalid_argument
        );
        std::size_t crc_pos = total_size - CRC32_SIZE;
        std::uint32_t expected_crc =
            std::to_integer<std::uint32_t>(base[crc_pos])
            | (std::to_integer<std::uint32_t>(base[crc_pos + 1]) << 8)
            | (std::to_integer<std::uint32_t>(base[crc_pos + 2]) << 16)
            | (std::to_integer<std::uint32_t>(base[crc_pos + 3]) << 24);
        std::uint32_t computed_crc =
            crc32_compute(base + header_size, crc_pos - header_size);
        RAPIDSMPF_EXPECTS(
            expected_crc == computed_crc,
            "CRC32 mismatch in serialized buffer",
            std::invalid_argument
        );
        data_limit = crc_pos;
    }
    return data_limit;
}
}  // namespace

extern char** environ;

namespace rapidsmpf::config {

Options::Options(std::unordered_map<std::string, OptionValue> options)
    : shared_{std::make_shared<detail::SharedOptions>()} {
    // insert, trim and lower case all keys.
    auto& opts = shared_->options;
    opts.reserve(options.size());
    for (auto&& [key, value] : options) {
        auto new_key = to_lower(trim(key));
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

bool Options::insert_if_absent(
    std::string const& key, std::string_view option_as_string
) {
    std::lock_guard<std::mutex> lock(shared_->mutex);
    auto [_, inserted] = shared_->options.try_emplace(
        to_lower(trim(key)), OptionValue{std::string(option_as_string)}
    );
    return inserted;
}

std::size_t Options::insert_if_absent(
    std::unordered_map<std::string, std::string> options_as_strings
) {
    std::size_t ret = 0;
    for (auto&& [key, val] : options_as_strings) {
        if (insert_if_absent(key, std::string_view{val})) {
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

    static_assert(
        MAX_OPTIONS <= std::numeric_limits<uint64_t>::max() / (2 * sizeof(uint64_t))
                           - sizeof(uint64_t),
        "MAX_OPTIONS too large, this will overflow header serialization"
    );

    RAPIDSMPF_EXPECTS(
        count <= MAX_OPTIONS, "too many options to serialize", std::invalid_argument
    );

    std::size_t const data_header_size = sizeof(uint64_t) + count * 2 * sizeof(uint64_t);
    std::size_t const header_size = PRELUDE_SIZE + data_header_size;

    std::size_t data_size = 0;
    for (auto const& [key, option] : shared.options) {
        RAPIDSMPF_EXPECTS(
            key.size() <= MAX_KEY_LEN,
            "key length exceeds maximum allowed size",
            std::invalid_argument
        );
        RAPIDSMPF_EXPECTS(
            !option.get_value().has_value(),
            "cannot serialize already parsed (accessed) option values",
            std::invalid_argument
        );
        auto const& val = option.get_value_as_string();
        RAPIDSMPF_EXPECTS(
            val.size() <= MAX_VALUE_LEN,
            "value length exceeds maximum allowed size",
            std::invalid_argument
        );
        data_size += key.size() + val.size();
    }
    RAPIDSMPF_EXPECTS(
        header_size + data_size + CRC32_SIZE <= MAX_TOTAL_SIZE,
        "serialized buffer exceeds maximum allowed size",
        std::invalid_argument
    );

    std::vector<std::uint8_t> buffer(header_size + data_size + CRC32_SIZE);
    auto base = reinterpret_cast<std::byte*>(buffer.data());

    // Write MAGIC and version prelude
    std::memcpy(base, MAGIC.data(), MAGIC.size());
    base[4] = FORMAT_VERSION;
    // flags: bit0 => CRC32 present
    base[5] = FLAG_CRC_PRESENT;
    base[6] = static_cast<std::byte>(0);
    base[7] = static_cast<std::byte>(0);

    // Write count (number of key-value pairs) after prelude.
    {
        auto const count_ = static_cast<uint64_t>(count);
        std::memcpy(base + PRELUDE_SIZE, &count_, sizeof(uint64_t));
    }

    // Prepare sorted entries by key for deterministic serialization
    std::vector<std::pair<std::string, std::string>> entries;
    entries.reserve(shared.options.size());
    for (auto const& kv : shared.options) {
        entries.emplace_back(kv.first, kv.second.get_value_as_string());
    }
    using entry_type = decltype(entries)::value_type;
    std::ranges::sort(entries, std::less{}, &entry_type::first);

    // Write offsets and data.
    std::size_t offset_index = 1;  // Offsets start after `count`.
    std::size_t data_offset = header_size;
    for (auto const& [key, value] : entries) {
        auto key_offset = static_cast<uint64_t>(data_offset);
        auto value_offset = static_cast<uint64_t>(key_offset + key.size());

        // Write offsets (placed after prelude + count)
        std::memcpy(
            base + PRELUDE_SIZE + offset_index * sizeof(uint64_t),
            &key_offset,
            sizeof(uint64_t)
        );
        std::memcpy(
            base + PRELUDE_SIZE + (offset_index + 1) * sizeof(uint64_t),
            &value_offset,
            sizeof(uint64_t)
        );
        offset_index += 2;

        // Write data
        std::memcpy(base + key_offset, key.data(), key.size());
        std::memcpy(base + value_offset, value.data(), value.size());

        data_offset = static_cast<std::size_t>(value_offset + value.size());
    }

    // Compute CRC32 over the data region and append at the end (little endian)
    {
        std::uint32_t crc = crc32_compute(base + header_size, data_size);
        std::size_t const crc_pos = header_size + data_size;
        base[crc_pos + 0] = static_cast<std::byte>(crc & 0xFFu);
        base[crc_pos + 1] = static_cast<std::byte>((crc >> 8) & 0xFFu);
        base[crc_pos + 2] = static_cast<std::byte>((crc >> 16) & 0xFFu);
        base[crc_pos + 3] = static_cast<std::byte>((crc >> 24) & 0xFFu);
    }

    return buffer;
}

Options Options::deserialize(std::vector<std::uint8_t> const& buffer) {
    auto const base = reinterpret_cast<std::byte const*>(buffer.data());
    std::size_t total_size = buffer.size();

    // Require MAGIC/version prelude
    RAPIDSMPF_EXPECTS(
        total_size >= PRELUDE_SIZE + sizeof(uint64_t),
        "buffer is too small to contain prelude and count",
        std::invalid_argument
    );
    RAPIDSMPF_EXPECTS(
        total_size <= MAX_TOTAL_SIZE,
        "serialized buffer exceeds maximum allowed size",
        std::invalid_argument
    );
    RAPIDSMPF_EXPECTS(
        std::memcmp(base, MAGIC.data(), MAGIC.size()) == 0,
        "buffer does not contain valid MAGIC",
        std::invalid_argument
    );
    uint64_t count = 0;
    std::byte version = base[4];
    std::byte flags = base[5];
    RAPIDSMPF_EXPECTS(
        version == FORMAT_VERSION,
        "unsupported Options serialization version",
        std::invalid_argument
    );
    std::memcpy(&count, base + PRELUDE_SIZE, sizeof(uint64_t));

    static_assert(
        MAX_OPTIONS <= std::numeric_limits<uint64_t>::max() / (2 * sizeof(uint64_t))
                           - sizeof(uint64_t),
        "MAX_OPTIONS too large, this will overflow header deserialization"
    );

    RAPIDSMPF_EXPECTS(
        count <= MAX_OPTIONS, "too many options to deserialize", std::invalid_argument
    );

    std::size_t const data_header_size = sizeof(uint64_t) + count * 2 * sizeof(uint64_t);
    std::size_t const header_size = PRELUDE_SIZE + data_header_size;
    RAPIDSMPF_EXPECTS(
        header_size <= total_size,
        "buffer is too small for header with declared count",
        std::invalid_argument
    );
    RAPIDSMPF_EXPECTS(
        static_cast<std::size_t>(count) <= MAX_OPTIONS,
        "too many options in serialized buffer",
        std::invalid_argument
    );

    std::size_t const data_limit =
        validate_crc_and_get_data_limit(base, header_size, total_size, flags);

    // Read offsets
    std::vector<uint64_t> key_offsets(count);
    std::vector<uint64_t> value_offsets(count);
    for (uint64_t i = 0; i < count; ++i) {
        std::memcpy(
            &key_offsets[i],
            base + PRELUDE_SIZE + (1 + 2 * i) * sizeof(uint64_t),
            sizeof(uint64_t)
        );
        std::memcpy(
            &value_offsets[i],
            base + PRELUDE_SIZE + (1 + 2 * i + 1) * sizeof(uint64_t),
            sizeof(uint64_t)
        );
    }

    // Reconstruct the key-value pairs with strict validation
    std::unordered_map<std::string, std::string> ret;
    for (uint64_t i = 0; i < count; ++i) {
        uint64_t const key_offset = key_offsets[i];
        uint64_t const value_offset = value_offsets[i];

        RAPIDSMPF_EXPECTS(
            key_offset >= header_size && value_offset >= header_size,
            "offsets must point to data region",
            std::out_of_range
        );
        RAPIDSMPF_EXPECTS(
            key_offset < value_offset,
            "key offset must be less than value offset",
            std::out_of_range
        );

        uint64_t const next_key_offset =
            (i + 1 < count) ? key_offsets[i + 1] : static_cast<uint64_t>(data_limit);
        RAPIDSMPF_EXPECTS(
            value_offset <= next_key_offset,
            "value data overlaps next key region",
            std::out_of_range
        );
        RAPIDSMPF_EXPECTS(
            key_offset < static_cast<uint64_t>(data_limit)
                && value_offset <= static_cast<uint64_t>(data_limit),
            "offsets exceed buffer size",
            std::out_of_range
        );

        auto const key_len = static_cast<std::size_t>(value_offset - key_offset);
        auto const value_len = static_cast<std::size_t>(next_key_offset - value_offset);
        RAPIDSMPF_EXPECTS(
            key_len <= MAX_KEY_LEN,
            "key length exceeds maximum allowed size",
            std::invalid_argument
        );
        RAPIDSMPF_EXPECTS(
            value_len <= MAX_VALUE_LEN,
            "value length exceeds maximum allowed size",
            std::invalid_argument
        );

        if (key_offset + key_len > data_limit || value_offset + value_len > data_limit) {
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
    // Compile anchored name pattern and avoid matching against the full "NAME=VALUE"
    std::regex name_pattern(
        "^" + key_regex + "$", std::regex::ECMAScript | std::regex::optimize
    );
    for (char** env = environ; *env != nullptr; ++env) {
        char const* cstr = *env;
        char const* eq = std::strchr(cstr, '=');
        if (!eq) {
            continue;
        }
        std::string name(cstr, static_cast<std::size_t>(eq - cstr));
        std::string value(eq + 1);
        std::smatch match;
        if (std::regex_match(name, match, name_pattern)) {
            if (match.size() == 2) {  // match[1]: captured key
                output.insert({match[1].str(), std::move(value)});
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
