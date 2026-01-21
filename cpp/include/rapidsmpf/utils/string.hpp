/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <array>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <string>

namespace rapidsmpf {

/**
 * @brief Formats value to a string with a specified number of decimal places.
 *
 * @tparam T The type of the value to format.
 * @param value The value to format.
 * @param precision The number of decimal places to include.
 * @return A string representation of the value with the specified precision.
 */
template <typename T>
std::string to_precision(T value, int precision = 2) {
    std::stringstream ss;
    ss.precision(precision);
    ss << std::fixed;
    ss << value;
    return ss.str();
}

/**
 * @brief Format number of bytes to a human readable string representation.
 *
 * @param nbytes The number of bytes to convert.
 * @param precision The number of decimal places to include.
 * @return A string representation of the byte size with the specified precision.
 */
std::string inline format_nbytes(double nbytes, int precision = 2) {
    constexpr std::array<const char*, 6> units = {" B", " KiB", " MiB", " GiB", " TiB"};
    double n = nbytes;
    for (auto const& unit : units) {
        if (std::abs(n) < 1024.0) {
            return to_precision(n, precision) + unit;
        }
        n /= 1024.0;
    }
    return to_precision(n, precision) + " PiB";
}

/**
 * @brief Format a time duration to a human-readable string representation.
 *
 * @param seconds The time duration to convert (in seconds).
 * @param precision The number of decimal places to include.
 * @return A string representation of the time duration with the specified precision.
 */
std::string inline format_duration(double seconds, int precision = 2) {
    double sec = std::abs(seconds);
    if (sec < 1e-6) {
        return to_precision(seconds * 1e9, precision) + " ns";
    } else if (sec < 1e-3) {
        return to_precision(seconds * 1e6, precision) + " us";
    } else if (sec < 1) {
        return to_precision(seconds * 1e3, precision) + " ms";
    } else {
        return to_precision(seconds, precision) + " s";
    }
}

/**
 * @brief Trims whitespace from both ends of the specified string.
 *
 * @param str The input string to be processed.
 * @return The trimmed string.
 */
std::string trim(std::string const& str);

/**
 * @brief Converts the specified string to lowercase.
 *
 * @param str The input string to be processed.
 * @return The trimmed string.
 */
std::string to_lower(std::string str);

/**
 * @brief Converts the specified string to uppercase.
 *
 * @param str The input string to be processed.
 * @return The trimmed string.
 */
std::string to_upper(std::string str);

/**
 * @brief Parses a string into a value of type T.
 *
 * This function attempts to parse the given string into a value of the specified
 * type `T` using a `std::stringstream`. If the parsing fails, an exception is thrown.
 *
 * @tparam T The type to parse the string into. Must support extraction from
 * `std::istream` via `operator>>`.
 * @param value The input string to parse.
 * @return T The parsed value of type `T`.
 *
 * @throws std::invalid_argument If the string cannot be parsed into the requested type.
 *
 * @note This function assumes that the input string contains a valid representation of
 * type `T`, and that `T` has a suitable `operator>>` overload.
 *
 * @example
 * int i = parse_string<int>("42");            // i == 42
 * double d = parse_string<double>("3.14");    // d == 3.14
 */
template <typename T>
T parse_string(std::string const& value) {
    std::stringstream sstream(value);
    T ret;
    sstream >> ret;
    if (sstream.fail()) {
        throw std::invalid_argument("cannot parse \"" + std::string{value} + "\"");
    }
    return ret;
}

/**
 * @brief Specialization of `parse_string` for boolean values.
 *
 * Converts the input string to a boolean. This function handles common boolean
 * representations such as `true`, `false`, `on`, `off`, `yes`, and `no`, as well as
 * numeric representations (e.g., `0` or `1`). The input is first checked for a numeric
 * value using `std::stoi`; if that fails, it is lowercased and trimmed before matching
 * against known textual representations.
 *
 * @param value String to convert to a boolean.
 * @return The corresponding boolean value.
 *
 * @throws std::invalid_argument If the string cannot be interpreted as a boolean.
 */
template <>
bool parse_string(std::string const& value);

}  // namespace rapidsmpf
