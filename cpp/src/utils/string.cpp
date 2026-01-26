/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <iomanip>
#include <optional>
#include <ranges>
#include <regex>

#include <rapidsmpf/utils/string.hpp>

namespace rapidsmpf {

std::string trim(std::string const& str) {
    std::stringstream trimmer;
    trimmer << str;
    std::string ret;
    trimmer >> ret;
    return ret;
}

std::string to_lower(std::string str) {
    // Special considerations regarding the case conversion:
    // - std::tolower() is not an addressable function. Passing it to std::transform()
    //   as a function pointer, if the compile turns out successful, causes the program
    //   behavior "unspecified (possibly ill-formed)", hence the lambda. ::tolower() is
    //   addressable and does not have this problem, but the following item still applies.
    // - To avoid UB in std::tolower() or ::tolower(), the character must be cast to
    // unsigned char.
    std::ranges::transform(str, str.begin(), [](unsigned char c) {
        return std::tolower(c);
    });
    return str;
}

std::string to_upper(std::string str) {
    // Special considerations regarding the case conversion, see to_lower().
    std::ranges::transform(str, str.begin(), [](unsigned char c) {
        return std::toupper(c);
    });
    return str;
}

namespace {

/**
 * @brief Remove the fractional part of a formatted number if it consists only of zeros.
 *
 * Examples:
 *   "10.00"  -> "10"
 *   "-1.000" -> "-1"
 *   "1.50"   -> unchanged
 */
std::string do_trim_zero_fraction(std::string const& value) {
    static const std::regex k_zero_fraction_regex(R"(^(-?\d+)\.0+$)");
    return std::regex_replace(value, k_zero_fraction_regex, "$1");
}

}  // namespace

std::string format_nbytes(
    double nbytes, int num_decimals, TrimZeroFraction trim_zero_fraction
) {
    constexpr std::array<char const*, 9> units{
        "B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB"
    };

    double size = std::abs(nbytes);
    std::size_t unit_idx = 0;

    if (std::isfinite(size)) {
        while (size >= 1024.0 && unit_idx + 1 < units.size()) {
            size /= 1024.0;
            ++unit_idx;
        }
    }

    std::ostringstream oss;
    if (nbytes < 0) {
        oss << '-';
    }
    oss << std::fixed << std::setprecision(num_decimals) << size;

    std::string ret = oss.str();
    if (trim_zero_fraction == TrimZeroFraction::YES && num_decimals > 0) {
        ret = do_trim_zero_fraction(ret);
    }

    ret += ' ';
    ret += units[unit_idx];
    return ret;
}

std::string format_duration(
    double seconds, int precision, TrimZeroFraction trim_zero_fraction
) {
    struct Unit {
        char const* name;
        double scale;
    };

    constexpr std::array<Unit, 3> large_units{{
        {.name = "d", .scale = 86400.0},
        {.name = "h", .scale = 3600.0},
        {.name = "min", .scale = 60.0},
    }};

    constexpr std::array<Unit, 4> subsecond_units{{
        {.name = "s", .scale = 1.0},
        {.name = "ms", .scale = 1e-3},
        {.name = "µs", .scale = 1e-6},
        {.name = "ns", .scale = 1e-9},
    }};

    double value = std::abs(seconds);
    char const* unit = "s";

    if (std::isfinite(value)) {
        for (const auto& u : large_units) {
            if (value >= u.scale) {
                value /= u.scale;
                unit = u.name;
                break;
            }
        }

        if (value < 1.0) {
            for (const auto& u : subsecond_units) {
                if (value >= u.scale) {
                    value /= u.scale;
                    unit = u.name;
                    break;
                }
            }
        }
    }

    std::ostringstream oss;
    if (seconds < 0) {
        oss << '-';
    }
    oss << std::fixed << std::setprecision(precision) << value;

    std::string ret = oss.str();
    if (trim_zero_fraction == TrimZeroFraction::YES && precision > 0) {
        ret = do_trim_zero_fraction(ret);
    }

    ret += ' ';
    ret += unit;
    return ret;
}

std::int64_t parse_nbytes(std::string_view text) {
    // Regex for parsing a human-readable byte count.
    //  - Group 1: signed floating-point number
    //      * integer or decimal form (e.g. "10", "1.5", ".5")
    //      * optional scientific notation (e.g. "1e6", "2.5E-3")
    //  - Group 2 (optional): unit suffix (e.g. "B", "KiB", "MiB", ...)
    //  - Leading and trailing whitespace is ignored
    //  - If no unit is present, the value is interpreted as bytes
    static const std::regex k_re(
        R"(^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-z]+)?\s*$)",
        std::regex::ECMAScript
    );

    std::cmatch m;
    if (!std::regex_match(text.begin(), text.end(), m, k_re)) {
        throw std::invalid_argument("invalid format: '" + std::string(text) + "'");
    }

    // Parse numeric part
    double value = 0.0;
    try {
        value = std::stod(m[1].str());
    } catch (const std::invalid_argument&) {
        throw std::invalid_argument("invalid number: '" + std::string(text) + "'");
    } catch (const std::out_of_range&) {
        throw std::out_of_range("number out of range: '" + std::string(text) + "'");
    }

    // Parse and normalize the unit suffix.
    //
    // Supported formats:
    //   - IEC (base-1024): KiB, MiB, GiB, TiB, PiB, EiB, ZiB, YiB
    //   - SI  (base-1000): KB,  MB,  GB,  TB,  PB,  EB,  ZB,  YB
    //   - Bytes: B
    //
    // Rules:
    //   - Units are case-insensitive.
    //   - Presence of 'I' selects IEC (base-1024); absence selects SI (base-1000).
    //   - If no unit is provided, the value is interpreted as bytes.
    //   - Any unrecognized unit results in std::invalid_argument.
    double multiplier = 1.0;  // default: bytes
    if (m[2].matched) {
        std::string unit = to_upper(m[2].str());

        // Special case: bytes
        if (unit == "B") {
            multiplier = 1.0;
        } else {
            // Match: K/M/G/T/P/E/Z/Y + optional 'I' + 'B'
            static const std::regex unit_re(R"(^([KMGTPEZY])(I)?B$)");

            std::cmatch um;
            if (!std::regex_match(unit.c_str(), um, unit_re)) {
                throw std::invalid_argument(
                    "unknown unit '" + unit + "' in '" + std::string(text) + "'"
                );
            }

            char const prefix = um[1].str()[0];
            bool const is_iec = um[2].matched;

            // Exponent by prefix position
            constexpr std::string_view prefixes = "KMGTPEZY";
            const auto pos = prefixes.find(prefix);
            if (pos == std::string_view::npos) {
                throw std::invalid_argument(
                    "unknown unit '" + unit + "' in '" + std::string(text) + "'"
                );
            }

            double const base = is_iec ? 1024.0 : 1000.0;
            multiplier = std::pow(base, static_cast<int>(pos) + 1);
        }
    }

    double const nbytes = value * multiplier;
    if (!std::isfinite(nbytes)) {
        throw std::out_of_range("non-finite result from '" + std::string(text) + "'");
    }

    double const rounded = std::llround(nbytes);
    if (rounded < static_cast<double>(std::numeric_limits<std::int64_t>::min())
        || rounded > static_cast<double>(std::numeric_limits<std::int64_t>::max()))
    {
        throw std::out_of_range("result out of range: '" + std::string(text) + "'");
    }

    return static_cast<std::int64_t>(rounded);
}

std::size_t parse_nbytes_unsigned(std::string_view text) {
    std::int64_t const value = parse_nbytes(text);

    if (value < 0) {
        throw std::invalid_argument(
            "negative value not allowed: '" + std::string(text) + "'"
        );
    }

    // Check against size_t range explicitly (important on 32-bit platforms).
    if (static_cast<std::uint64_t>(value)
        > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    {
        throw std::out_of_range(
            "value out of range for size_t: '" + std::string(text) + "'"
        );
    }

    return static_cast<std::size_t>(value);
}

std::size_t parse_nbytes_or_percent(std::string_view text, double total_bytes) {
    if (total_bytes <= 0.0) {
        throw std::invalid_argument("total_bytes must be positive");
    }

    static const std::regex k_re(
        R"(^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-z]+|%)?\s*$)",
        std::regex::ECMAScript
    );

    std::cmatch m;
    if (!std::regex_match(text.begin(), text.end(), m, k_re)) {
        throw std::invalid_argument("invalid format: '" + std::string(text) + "'");
    }

    std::string number = m[1].str();
    std::string suffix;
    if (m[2].matched) {
        suffix = m[2].str();
    }

    // Percentage case: interpret as percent of total_bytes.
    if (suffix == "%") {
        // parse_nbytes_unsigned validates and rounds the numeric token
        std::size_t const percent = parse_nbytes_unsigned(number);

        auto const scaled = (static_cast<double>(percent) * total_bytes) / 100.0;
        if (scaled > static_cast<double>(std::numeric_limits<std::size_t>::max())) {
            throw std::out_of_range("percent conversion exceeds std::size_t");
        }
        return static_cast<std::size_t>(scaled);
    }

    // Absolute byte case: interpret like parse_nbytes_unsigned
    if (!suffix.empty()) {
        number += suffix;
    }
    return parse_nbytes_unsigned(number);
}

Duration parse_duration(std::string_view text) {
    // Regex for parsing a human-readable duration.
    //  - Group 1: signed floating-point number
    //      * integer or decimal form (e.g. "10", "1.5", ".5")
    //      * optional scientific notation (e.g. "1e6", "2.5E-3")
    //  - Group 2 (optional): unit suffix
    //      * supported units: "ms", "s", "m" (minutes), "min", "h", "d"
    //  - Leading and trailing whitespace is ignored
    //  - If no unit is present, the value is interpreted as seconds
    static const std::regex k_re(
        R"(^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-zµ]+)?\s*$)",
        std::regex::ECMAScript
    );

    std::cmatch m;
    if (!std::regex_match(text.begin(), text.end(), m, k_re)) {
        throw std::invalid_argument("parse_duration: invalid format");
    }

    // Parse numeric part.
    double value = 0.0;
    try {
        value = std::stod(m[1].str());
    } catch (const std::invalid_argument&) {
        throw std::invalid_argument("parse_duration: invalid number");
    } catch (const std::out_of_range&) {
        throw std::out_of_range("parse_duration: number out of range");
    }

    // Default unit: seconds.
    double multiplier = 1.0;
    if (m[2].matched) {
        std::string unit = to_lower(m[2].str());
        if (unit == "ns")
            multiplier = 1e-9;
        else if (unit == "µs" || unit == "us")
            multiplier = 1e-6;
        else if (unit == "ms")
            multiplier = 1e-3;
        else if (unit == "s")
            multiplier = 1.0;
        else if (unit == "m" || unit == "min")
            multiplier = 60.0;
        else if (unit == "h")
            multiplier = 3600.0;
        else if (unit == "d")
            multiplier = 86400.0;
        else {
            throw std::invalid_argument("parse_duration: unknown unit");
        }
    }

    double const seconds = value * multiplier;
    if (!std::isfinite(seconds)) {
        throw std::out_of_range("parse_duration: non-finite result");
    }
    return Duration{seconds};
}

template <>
bool parse_string(std::string const& text) {
    try {
        // Try parsing `text` as a integer.
        return static_cast<bool>(std::stoi(text));
    } catch (std::invalid_argument const&) {
    }
    std::string str = to_lower(trim(text));
    if (str == "true" || str == "on" || str == "yes") {
        return true;
    }
    if (str == "false" || str == "off" || str == "no") {
        return false;
    }
    throw std::invalid_argument("cannot parse \"" + std::string{text} + "\"");
}

std::optional<std::string> parse_optional(std::string text) {
    static const std::regex disabled_re(
        R"(^\s*(false|no|off|disable|disabled|none|n/a|na)\s*$)",
        std::regex::ECMAScript | std::regex::icase
    );
    if (std::regex_match(text, disabled_re)) {
        return std::nullopt;
    }
    return text;
}

}  // namespace rapidsmpf
