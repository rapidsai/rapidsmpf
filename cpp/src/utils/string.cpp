/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <iomanip>
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
        const char* name;
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
    const char* unit = "s";

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
        throw std::invalid_argument("parse_nbytes: invalid format");
    }

    // Parse numeric part
    double value = 0.0;
    try {
        value = std::stod(m[1].str());
    } catch (const std::invalid_argument&) {
        throw std::invalid_argument("parse_nbytes: invalid number");
    } catch (const std::out_of_range&) {
        throw std::out_of_range("parse_nbytes: number out of range");
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
        std::string unit = m[2].str();

        // Normalize case for simpler matching
        for (char& c : unit) {
            c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        }

        // Special case: bytes
        if (unit == "B") {
            multiplier = 1.0;
        } else {
            // Match: K/M/G/T/P/E/Z/Y + optional 'I' + 'B'
            static const std::regex unit_re(R"(^([KMGTPEZY])(I)?B$)");

            std::cmatch um;
            if (!std::regex_match(unit.c_str(), um, unit_re)) {
                throw std::invalid_argument("parse_nbytes: unknown unit");
            }

            const char prefix = um[1].str()[0];
            const bool is_iec = um[2].matched;

            // Exponent by prefix position
            constexpr std::string_view prefixes = "KMGTPEZY";
            const auto pos = prefixes.find(prefix);
            if (pos == std::string_view::npos) {
                throw std::invalid_argument("parse_nbytes: unknown unit");
            }

            const double base = is_iec ? 1024.0 : 1000.0;
            multiplier = std::pow(base, static_cast<int>(pos) + 1);
        }
    }

    const double nbytes = value * multiplier;
    if (!std::isfinite(nbytes)) {
        throw std::out_of_range("parse_nbytes: non-finite result");
    }

    const double rounded = std::llround(nbytes);
    if (rounded < static_cast<double>(std::numeric_limits<std::int64_t>::min())
        || rounded > static_cast<double>(std::numeric_limits<std::int64_t>::max()))
    {
        throw std::out_of_range("parse_nbytes: result out of range");
    }
    return static_cast<std::int64_t>(rounded);
}

template <>
bool parse_string(std::string const& value) {
    try {
        // Try parsing `value` as a integer.
        return static_cast<bool>(std::stoi(value));
    } catch (std::invalid_argument const&) {
    }
    std::string str = to_lower(trim(value));
    if (str == "true" || str == "on" || str == "yes") {
        return true;
    }
    if (str == "false" || str == "off" || str == "no") {
        return false;
    }
    throw std::invalid_argument("cannot parse \"" + std::string{value} + "\"");
}

}  // namespace rapidsmpf
