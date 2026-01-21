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

std::string format_nbytes(
    double nbytes, int num_decimals, TrimZeroFraction trim_zero_fraction
) {
    constexpr std::array<const char*, 9> units{
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
        static const std::regex k_zero_fraction_regex(R"(^(-?\d+)\.0+$)");
        ret = std::regex_replace(ret, k_zero_fraction_regex, "$1");
    }

    ret += ' ';
    ret += units[unit_idx];
    return ret;
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
