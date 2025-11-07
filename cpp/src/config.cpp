/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
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
#include <rapidsmpf/utils.hpp>

namespace {
// Serialization limits and format configuration (implementation details)
inline constexpr std::size_t MAX_OPTIONS = 65536;
inline constexpr std::size_t MAX_KEY_LEN = 4 * 1024;
inline constexpr std::size_t MAX_VALUE_LEN = 1 * 1024 * 1024;
inline constexpr std::size_t MAX_TOTAL_SIZE = 64 * 1024 * 1024;

// Format constants
inline constexpr std::array<std::uint8_t, 4> MAGIC{{'R', 'M', 'P', 'F'}};
inline constexpr std::uint8_t FORMAT_VERSION = 1;
inline constexpr std::uint8_t FLAG_CRC_PRESENT = 0x01;
inline constexpr std::size_t PRELUDE_SIZE_BYTES =
    8;  // MAGIC(4) + version(1) + flags(1) + reserved(2)

// Simple CRC32 (IEEE 802.3) without table for compactness
inline std::uint32_t crc32_compute(const std::uint8_t* data, std::size_t length) {
    std::uint32_t crc = 0xFFFFFFFFu;
    for (std::size_t i = 0; i < length; ++i) {
        crc ^= static_cast<std::uint32_t>(data[i]);
        for (int bit = 0; bit < 8; ++bit) {
            std::uint32_t mask = -(crc & 1u);
            crc = (crc >> 1) ^ (0xEDB88320u & mask);
        }
    }
    return ~crc;
}

inline bool checked_add_u64(std::uint64_t a, std::uint64_t b, std::uint64_t* out) {
    if (out == nullptr)
        return false;
    if (std::numeric_limits<std::uint64_t>::max() - a < b) {
        return false;
    }
    *out = a + b;
    return true;
}

inline bool checked_mul_u64(std::uint64_t a, std::uint64_t b, std::uint64_t* out) {
    if (out == nullptr)
        return false;
    if (a == 0 || b == 0) {
        *out = 0;
        return true;
    }
    if (a > std::numeric_limits<std::uint64_t>::max() / b) {
        return false;
    }
    *out = a * b;
    return true;
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
    RAPIDSMPF_EXPECTS(
        count <= MAX_OPTIONS, "too many options to serialize", std::invalid_argument
    );
    // Header format:
    // - prelude: [4-byte MAGIC "RMPF"][1-byte version][1-byte flags][2-byte (reserved)]
    // - data header: [uint64_t (count)][count * 2 * uint64_t (offset pairs)]
    // Use PRELUDE_SIZE_BYTES constant for positions
    std::size_t const prelude_size = PRELUDE_SIZE_BYTES;
    // Compute header size with checked arithmetic
    std::uint64_t pairs_u64 = 0;
    std::uint64_t offs_bytes_u64 = 0;
    std::uint64_t data_header_u64 = 0;
    bool ok = checked_mul_u64(static_cast<std::uint64_t>(count), 2ULL, &pairs_u64)
              && checked_mul_u64(pairs_u64, sizeof(uint64_t), &offs_bytes_u64)
              && checked_add_u64(sizeof(uint64_t), offs_bytes_u64, &data_header_u64);
    RAPIDSMPF_EXPECTS(
        ok && data_header_u64 <= std::numeric_limits<std::size_t>::max() - prelude_size,
        "header size overflow",
        std::invalid_argument
    );
    std::size_t const data_header_size = static_cast<std::size_t>(data_header_u64);
    std::size_t const header_size = prelude_size + data_header_size;

    std::size_t data_size = 0;
    for (auto const& [key, option] : shared.options) {
        RAPIDSMPF_EXPECTS(
            key.size() <= MAX_KEY_LEN,
            "key length exceeds maximum allowed size",
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
        prelude_size + data_header_size + data_size + 4 <= MAX_TOTAL_SIZE,
        "serialized buffer exceeds maximum allowed size",
        std::invalid_argument
    );

    // Allocate extra 4 bytes for CRC32 over the data region
    std::vector<std::uint8_t> buffer(header_size + data_size + 4);
    std::uint8_t* base = buffer.data();

    // Write MAGIC and version prelude
    std::memcpy(base, MAGIC.data(), MAGIC.size());
    base[4] = FORMAT_VERSION;
    // flags: bit0 => CRC32 present
    base[5] = FLAG_CRC_PRESENT;
    base[6] = 0;
    base[7] = 0;

    // Write count (number of key-value pairs) after prelude.
    {
        auto const count_ = static_cast<uint64_t>(count);
        std::memcpy(base + prelude_size, &count_, sizeof(uint64_t));
    }

    for (auto const& [key, option] : shared.options) {
        RAPIDSMPF_EXPECTS(
            !option.get_value().has_value(),
            "cannot serialize already parsed (accessed) option values",
            std::invalid_argument
        );
    }

    // Prepare sorted entries by key for deterministic serialization
    std::vector<std::pair<std::string, std::string>> entries;
    entries.reserve(shared.options.size());
    for (auto const& kv : shared.options) {
        entries.emplace_back(kv.first, kv.second.get_value_as_string());
    }
    std::sort(entries.begin(), entries.end(), [](auto const& a, auto const& b) {
        return a.first < b.first;
    });

    // Write offsets and data.
    std::size_t offset_index = 1;  // Offsets start after `count`.
    std::size_t data_offset = header_size;
    for (auto const& kv : entries) {
        auto const& key = kv.first;
        auto const& value = kv.second;

        auto key_offset = static_cast<uint64_t>(data_offset);
        auto value_offset = static_cast<uint64_t>(key_offset + key.size());

        // Write offsets (placed after prelude + count)
        std::memcpy(
            base + prelude_size + offset_index * sizeof(uint64_t),
            &key_offset,
            sizeof(uint64_t)
        );
        std::memcpy(
            base + prelude_size + (offset_index + 1) * sizeof(uint64_t),
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
        base[crc_pos + 0] = static_cast<std::uint8_t>(crc & 0xFFu);
        base[crc_pos + 1] = static_cast<std::uint8_t>((crc >> 8) & 0xFFu);
        base[crc_pos + 2] = static_cast<std::uint8_t>((crc >> 16) & 0xFFu);
        base[crc_pos + 3] = static_cast<std::uint8_t>((crc >> 24) & 0xFFu);
    }

    return buffer;
}

Options Options::deserialize(std::vector<std::uint8_t> const& buffer) {
    const std::uint8_t* base = buffer.data();
    std::size_t total_size = buffer.size();

    // Require MAGIC/version prelude
    RAPIDSMPF_EXPECTS(
        total_size >= PRELUDE_SIZE_BYTES + sizeof(uint64_t)
            && std::memcmp(base, MAGIC.data(), MAGIC.size()) == 0,
        "buffer is too small to contain prelude and count",
        std::invalid_argument
    );
    uint64_t count = 0;
    std::size_t prelude_size = PRELUDE_SIZE_BYTES;  // MAGIC + version + flags/reserved
    std::uint8_t version = base[4];
    std::uint8_t flags = base[5];
    RAPIDSMPF_EXPECTS(
        version == 1, "unsupported Options serialization version", std::invalid_argument
    );
    std::memcpy(&count, base + prelude_size, sizeof(uint64_t));

    // Compute header size with checked arithmetic and enforce limits
    std::uint64_t pairs_u64 = 0;
    std::uint64_t offs_bytes_u64 = 0;
    std::uint64_t data_header_u64 = 0;
    bool ok = checked_mul_u64(static_cast<std::uint64_t>(count), 2ULL, &pairs_u64)
              && checked_mul_u64(pairs_u64, sizeof(uint64_t), &offs_bytes_u64)
              && checked_add_u64(sizeof(uint64_t), offs_bytes_u64, &data_header_u64);
    RAPIDSMPF_EXPECTS(
        ok && data_header_u64 <= std::numeric_limits<std::size_t>::max() - prelude_size,
        "header size overflow",
        std::invalid_argument
    );
    std::size_t const data_header_size = static_cast<std::size_t>(data_header_u64);
    std::size_t const header_size = prelude_size + data_header_size;
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
    RAPIDSMPF_EXPECTS(
        total_size <= MAX_TOTAL_SIZE,
        "serialized buffer exceeds maximum allowed size",
        std::invalid_argument
    );

    // If CRC32 is present (v1 flag), validate it and adjust data limit
    std::size_t data_limit = total_size;
    if ((flags & 0x01u) != 0u) {
        RAPIDSMPF_EXPECTS(
            total_size >= header_size + 4,
            "buffer too small for CRC32 trailer",
            std::invalid_argument
        );
        std::size_t crc_pos = total_size - 4;
        std::uint32_t expected_crc =
            static_cast<std::uint32_t>(base[crc_pos])
            | (static_cast<std::uint32_t>(base[crc_pos + 1]) << 8)
            | (static_cast<std::uint32_t>(base[crc_pos + 2]) << 16)
            | (static_cast<std::uint32_t>(base[crc_pos + 3]) << 24);
        std::uint32_t computed_crc =
            crc32_compute(base + header_size, crc_pos - header_size);
        RAPIDSMPF_EXPECTS(
            expected_crc == computed_crc,
            "CRC32 mismatch in serialized buffer",
            std::invalid_argument
        );
        data_limit = crc_pos;
    }

    // Read offsets
    std::vector<uint64_t> key_offsets(count);
    std::vector<uint64_t> value_offsets(count);
    for (uint64_t i = 0; i < count; ++i) {
        std::memcpy(
            &key_offsets[i],
            base + prelude_size + (1 + 2 * i) * sizeof(uint64_t),
            sizeof(uint64_t)
        );
        std::memcpy(
            &value_offsets[i],
            base + prelude_size + (1 + 2 * i + 1) * sizeof(uint64_t),
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

        std::size_t const key_len = static_cast<std::size_t>(value_offset - key_offset);
        std::size_t const value_len =
            static_cast<std::size_t>(next_key_offset - value_offset);
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
        const char* cstr = *env;
        const char* eq = std::strchr(cstr, '=');
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
