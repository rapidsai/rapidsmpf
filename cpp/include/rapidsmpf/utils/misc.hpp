/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <memory>
#include <ranges>
#include <source_location>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

namespace rapidsmpf {

/** @brief Helper macro to check if the CUDA version is at least the specified version.
 *
 * @param version The minimum CUDA version to check against. Must be in the format of
 * MAJOR*1000 + MINOR*10.
 */
#define RAPIDSMPF_CUDA_VERSION_AT_LEAST(version) (CUDART_VERSION >= version)

/// Alias for high-resolution clock from the chrono library.
using Clock = std::chrono::high_resolution_clock;
/// Alias for a duration type representing time in seconds as a double.
using Duration = std::chrono::duration<double>;

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
 * @brief Converts a map-like associative container to a vector by moving the values and
 * discarding the keys.
 *
 * @tparam MapType The type of the map-like associative container. Must provide a
 * mapped_type and support range-based for-loops.
 * @param map The map whose values will be moved into the resulting vector. Keys are
 * ignored.
 * @returns A std::vector containing the moved values from the input map.
 */
template <typename MapType>
auto to_vector(MapType&& map) {
    using ValueType = typename std::remove_reference_t<MapType>::mapped_type;
    std::vector<ValueType> vec;
    vec.reserve(map.size());
    for (auto&& [key, value] : map) {
        vec.push_back(std::move(value));
    }
    return vec;
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

/// @brief Helper for overloaded lambdas using std::visit.
template <class... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};

/**
 * @brief Backport of `std::ranges::contains` from C++23 for C++20.
 *
 * Checks whether a range contains a given value.
 *
 * @tparam R An input range type.
 * @tparam T The type of the value to search for.
 * @tparam Proj A projection function applied to each element before comparison.
 *
 * @param range The range to search.
 * @param value The value to search for in the range.
 * @param proj  The projection to apply to each element before comparison.
 *
 * @return true if any element in the range compares equal to value after projection,
 *         false otherwise.
 */
template <std::ranges::input_range R, typename T, typename Proj = std::identity>
[[nodiscard]] constexpr bool contains(R&& range, T const& value, Proj proj = {}) {
    for (auto const& elem : range) {
        if (std::invoke(proj, elem) == value) {
            return true;
        }
    }
    return false;
}

/**
 * @brief Safely casts a numeric value to another type with overflow checking.
 *
 * @tparam To The destination type.
 * @tparam From The source type.
 * @param value The value to cast.
 * @param loc Source location (automatically captured).
 * @return To The safely cast value.
 *
 * @throws std::overflow_error if the value cannot be represented in the destination type.
 */
template <typename To, typename From>
  requires std::is_arithmetic_v<To> && std::is_arithmetic_v<From>
constexpr To safe_cast(
    From value, std::source_location const& loc = std::source_location::current()
) {

    if constexpr (std::is_same_v<From, To>) {
        // Same type, no-op.
        return value;
    } else if constexpr (std::is_integral_v<From> && std::is_integral_v<To>) {
        // Integer to integer.
        if (!std::in_range<To>(value)) {
            throw std::overflow_error(
                "RapidsMPF cast error at: " + std::string(loc.file_name()) + ":"
                + std::to_string(loc.line()) + ", value out of range"
            );
        }
        return static_cast<To>(value);
    } else {
        // Floating point conversions: direct cast (well-defined overflow behavior).
        return static_cast<To>(value);
    }
}

}  // namespace rapidsmpf
