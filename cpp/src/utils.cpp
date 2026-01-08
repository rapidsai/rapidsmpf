/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <ranges>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/utils.hpp>

namespace rapidsmpf {

#if __has_include(<valgrind/valgrind.h>)
#include <valgrind/valgrind.h>

bool is_running_under_valgrind() {
    static bool ret = RUNNING_ON_VALGRIND;
    return ret;
}
#else
bool is_running_under_valgrind() {
    return false;
}
#endif

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
