/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <string>

namespace rapidsmpf {

/// Alias for high-resolution clock from the chrono library.
using Clock = std::chrono::high_resolution_clock;
/// Alias for a duration type representing time in seconds as a double.
using Duration = std::chrono::duration<double>;

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
 * @brief Extracts a key-value pair from a map, removing it from the map.
 *
 * @tparam MapType The type of the associative container.
 * @param map The map from which to extract the key-value pair.
 * @param position Const iterator pointing to a node in the map.
 * @return A pair containing the extracted key and value.
 *
 * @note Invalidates any iterators to the extracted element (notably `position`).
 *
 * @throws std::out_of_range If the iterator is not found in the map.
 */
template <typename MapType>
std::pair<typename MapType::key_type, typename MapType::mapped_type> extract_item(
    MapType& map, typename MapType::const_iterator position
) {
    auto node = map.extract(position);
    if (!node) {
        throw std::out_of_range("Invalid iterator passed to extract");
    }
    return {std::move(node.key()), std::move(node.mapped())};
}

/**
 * @brief Extracts a key-value pair from a map, removing it from the map.
 *
 * @tparam MapType The type of the associative container.
 * @param map The map from which to extract the key-value pair.
 * @param key The key to extract.
 * @return A pair containing the extracted key and value.
 *
 * @throws std::out_of_range If the key is not found in the map.
 */
template <typename MapType>
std::pair<typename MapType::key_type, typename MapType::mapped_type> extract_item(
    MapType& map, typename MapType::key_type const& key
) {
    auto node = map.extract(key);
    if (!node) {
        throw std::out_of_range("Invalid key passed to extract");
    }
    return {std::move(node.key()), std::move(node.mapped())};
}

/**
 * @brief Extracts the value associated with a specific key from a map, removing the
 * key-value pair.
 *
 * @tparam MapType The type of the associative container.
 * @param map The map from which to extract the value.
 * @param key The key associated with the value to extract.
 * @return The extracted value.
 *
 * @throws std::out_of_range If the key is not found in the map.
 */
template <typename MapType>
typename MapType::mapped_type extract_value(
    MapType& map, typename MapType::key_type const& key
) {
    return std::move(extract_item(map, key).second);
}

/**
 * @brief Extracts the value associated with a specific key from a map, removing the
 * key-value pair.
 *
 * @tparam MapType The type of the associative container.
 * @param map The map from which to extract the value.
 * @param position Const iterator pointing to a node in the map.
 * @return The extracted value.
 *
 * @note Invalidates any iterators to the extracted element (notably `position`).
 *
 * @throws std::out_of_range If the key is not found in the map.
 */
template <typename MapType>
typename MapType::mapped_type extract_value(
    MapType& map, typename MapType::const_iterator position
) {
    return std::move(extract_item(map, position).second);
}

/**
 * @brief Extracts a key from a map, removing the key-value pair.
 *
 * @tparam MapType The type of the associative container.
 * @param map The map from which to extract the key.
 * @param key The key to extract.
 * @return The extracted key.
 *
 * @throws std::out_of_range If the key is not found in the map.
 */
template <typename MapType>
typename MapType::key_type extract_key(
    MapType& map, typename MapType::key_type const& key
) {
    return std::move(extract_item(map, key).first);
}

/**
 * @brief Extracts a key from a map, removing the key-value pair.
 *
 * @tparam MapType The type of the associative container.
 * @param map The map from which to extract the key.
 * @param position Const iterator pointing to a node in the map.
 * @return The extracted key.
 *
 * @note Invalidates any iterators to the extracted element (notably `position`).
 *
 * @throws std::out_of_range If the key is not found in the map.
 */
template <typename MapType>
typename MapType::key_type extract_key(
    MapType& map, typename MapType::const_iterator position
) {
    return std::move(extract_item(map, position).first);
}

/**
 * @brief Checks whether the application is running under Valgrind.
 *
 * @return `true` if the application is running under Valgrind, `false` otherwise.
 */
bool is_running_under_valgrind();

/**
 * @brief Performs safe division, returning 0 if the denominator is zero.
 *
 * @tparam T The numeric type of the operands.
 * @param x The numerator.
 * @param y The denominator.
 * @return T The result of x / y, or 0 if y is zero.
 */
template <typename T>
constexpr T safe_div(T x, T y) {
    return (y == 0) ? 0 : x / y;
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

// Macro to concatenate two tokens x and y.
#define RAPIDSMPF_CONCAT_DETAIL_(x, y) x##y
#define RAPIDSMPF_CONCAT(x, y) RAPIDSMPF_CONCAT_DETAIL_(x, y)

// Stringify a macro argument.
#define RAPIDSMPF_STRINGIFY_DETAIL_(x) #x
#define RAPIDSMPF_STRINGIFY(x) RAPIDSMPF_STRINGIFY_DETAIL_(x)

/**
 * @def RAPIDSMPF_OVERLOAD_BY_ARG_COUNT
 * @brief Helper macro to select another macro based on the number of arguments.
 *
 * Example usage:
 * @code
 * #define FOO_1(x)        do_something_with_one(x)
 * #define FOO_2(x, y)     do_something_with_two(x, y)
 *
 * #define FOO(...) RAPIDSMPF_OVERLOAD_BY_ARG_COUNT \
 *                      (__VA_ARGS__, FOO_2, FOO_1)(__VA_ARGS__)
 *
 * FOO(42);        // Expands to FOO_1(42)
 * FOO(1, 2);      // Expands to FOO_2(1, 2)
 * @endcode
 */
#define RAPIDSMPF_OVERLOAD_BY_ARG_COUNT(_1, _2, NAME, ...) NAME

namespace detail {

/**
 * @brief Returns the raw pointer from a pointer, reference, or smart pointer.
 *
 * This utility is useful in macros that accepts any kind of reference.
 *
 * @tparam T Type of the object.
 * @param ptr A raw pointer.
 * @return T* The same raw pointer.
 */
template <typename T>
constexpr T* to_pointer(T* ptr) noexcept {
    return ptr;
}

/** @copydoc to_pointer(T*) */
template <typename T>
constexpr T* to_pointer(T& ptr) noexcept {
    return std::addressof(ptr);
}

/** @copydoc to_pointer(T*) */
template <typename T>
constexpr T* to_pointer(std::unique_ptr<T>& ptr) noexcept {
    return ptr.get();
}

/** @copydoc to_pointer(T*) */
template <typename T>
constexpr T* to_pointer(std::shared_ptr<T>& ptr) noexcept {
    return ptr.get();
}

}  // namespace detail


}  // namespace rapidsmpf
