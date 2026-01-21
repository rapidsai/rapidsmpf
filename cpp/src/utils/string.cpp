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
    constexpr std::array<char const *, 9> units{
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

namespace {

double unit_multiplier(std::string_view unit)
{
    if (unit.empty()) {
        return 1.0;  // default: bytes
    }

    constexpr std::array<std::pair<std::string_view, int>, 9> k_units{{
        {"B", 0},   {"KiB", 1}, {"MiB", 2}, {"GiB", 3}, {"TiB", 4},
        {"PiB", 5}, {"EiB", 6}, {"ZiB", 7}, {"YiB", 8},
    }};

    for (const auto& [name, pow] : k_units) {
        if (unit.size() == name.size()) {
            bool match = true;
            for (std::size_t i = 0; i < unit.size(); ++i) {
                auto a = static_cast<unsigned char>(unit[i]);
                auto b = static_cast<unsigned char>(name[i]);
                if (std::tolower(a) != std::tolower(b)) {
                    match = false;
                    break;
                }
            }
            if (match) {
                return std::ldexp(1.0, 10 * pow);  // 1024^pow
            }
        }
    }

    throw std::invalid_argument("parse_nbytes: unknown unit");
}

}  // namespace

std::int64_t parse_nbytes(std::string_view text)
{
    // 1: number, 2: unit (optional)
    static const std::regex k_re(
        R"(^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*([A-Za-z]+)?\s*$)",
        std::regex::ECMAScript);

    std::cmatch m;
    if (!std::regex_match(text.begin(), text.end(), m, k_re)) {
        throw std::invalid_argument("parse_nbytes: invalid format");
    }

    const std::string number_str = m[1].str();
    const std::string unit_str = m[2].matched ? m[2].str() : std::string{};

    double value = 0.0;
    try {
        value = std::stod(number_str);
    } catch (const std::invalid_argument&) {
        throw std::invalid_argument("parse_nbytes: invalid number");
    } catch (const std::out_of_range&) {
        throw std::out_of_range("parse_nbytes: number out of range");
    }

    const double mult = unit_multiplier(unit_str);
    const double bytes_d = value * mult;

    if (!std::isfinite(bytes_d)) {
        throw std::out_of_range("parse_nbytes: non-finite result");
    }

    const double rounded = std::llround(bytes_d);

    if (rounded < static_cast<double>(std::numeric_limits<std::int64_t>::min()) ||
        rounded > static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
        throw std::out_of_range("parse_nbytes: result out of int64 range");
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
