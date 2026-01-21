/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <array>
#include <cmath>
#include <cstdint>
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

/// @brief Control whether a zero fractional part is omitted when formatting values.
enum class TrimZeroFraction {
    NO,  ///< Always keep the fractional part.
    YES,  ///< Omit the fractional part when it consists only of zeros.
};

/**
 * @brief Format a byte count as a human-readable string using IEC units.
 *
 * Converts an integer byte count into a scaled string representation using
 * binary (base-1024) units such as KiB, MiB, and GiB.
 *
 * Negative values are supported and are formatted with a leading minus sign,
 * which is useful when representing signed byte deltas or accounting values.
 *
 * Decimal formatting is controlled by @p precision. When @p trim_zero_fraction
 * is set to @c TrimZeroFraction::YES, the fractional part is omitted entirely
 * if all decimal digits are zero. Otherwise, the specified number of decimal
 * places is preserved.
 *
 * Examples:
 *   - 1024 bytes with 2 decimals → "1.00 KiB" or "1 KiB" (trimmed)
 *   - 1536 bytes with 2 decimals → "1.50 KiB"
 *
 * @param nbytes Signed number of bytes to format, provided as a double to support
 * any integer magnitude.
 * @param num_decimals Number of decimal places to include in the formatted value.
 * @param trim_zero_fraction Whether to omit the fractional part when it consists
 * only of zeros.
 * @return Human-readable string representation of the byte count.
 */
std::string format_nbytes(
    double nbytes,
    int num_decimals = 2,
    TrimZeroFraction trim_zero_fraction = TrimZeroFraction::YES
);

/**
 * @brief Format a time duration as a human-readable string.
 *
 * Converts a duration given in seconds into a scaled string representation
 * using common time units such as ns, µs, ms, s, min, h, and d.
 *
 * The duration is accepted as a @c double to support both fractional seconds
 * and very large values without overflow.
 *
 * Negative values are supported and are formatted with a leading minus sign,
 * which is useful when representing signed time deltas.
 *
 * Decimal formatting is controlled by @p precision. When @p trim_zero_fraction
 * is set to @c TrimZeroFraction::YES, the fractional part is omitted entirely
 * if all decimal digits are zero. Otherwise, the specified number of decimal
 * places is preserved.
 *
 * @param seconds Time duration to format, in seconds.
 * @param precision Number of decimal places to include in the formatted value.
 * @param trim_zero_fraction Whether to omit the fractional part when it consists only of
 * zeros.
 * @return Human-readable string representation of the time duration.
 */
std::string format_duration(
    double seconds,
    int precision = 2,
    TrimZeroFraction trim_zero_fraction = TrimZeroFraction::YES
);

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
