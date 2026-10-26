/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <algorithm>
#include <array>
#include <initializer_list>
#include <iostream>
#include <ranges>
#include <span>

#include <rapidsmpf/utils/misc.hpp>

namespace rapidsmpf {

/// @brief Enum representing the type of memory sorted in decreasing order of preference.
enum class MemoryType : int {
    DEVICE = 0,  ///< Device memory
    PINNED_HOST = 1,  ///< Pinned host memory
    HOST = 2,  ///< Host memory
    DISK = 3  ///< Disk-backed storage
};

/// @brief All memory types sorted in decreasing order of preference.
constexpr std::array MEMORY_TYPES{
    MemoryType::DEVICE, MemoryType::PINNED_HOST, MemoryType::HOST, MemoryType::DISK
};

/// @brief Memory type names sorted to match `MemoryType` and `MEMORY_TYPES`.
constexpr std::array<char const*, MEMORY_TYPES.size()> MEMORY_TYPE_NAMES{
    {"DEVICE", "PINNED_HOST", "HOST", "DISK"}
};

/**
 * @brief Get the memory types with preference between @p from and @p to, inclusive.
 *
 * The returned span is a contiguous sub-range `[from, to]` of @p mem_types, which must
 * list memory types in decreasing order of preference.
 *
 * @param from The most preferred memory type of the range.
 * @param to The least preferred memory type of the range.
 * @param mem_types The memory types to select from, `MEMORY_TYPES` by default.
 * @return A span of memory types from @p from to @p to, empty if @p from is not in
 * @p mem_types, @p to does not come at or after @p from, or either of them isn't a valid
 * `MemoryType`.
 */
constexpr std::span<MemoryType const> preference_between(
    MemoryType from, MemoryType to, std::span<MemoryType const> mem_types = MEMORY_TYPES
) noexcept {
    if (!contains(MEMORY_TYPES, from) || !contains(MEMORY_TYPES, to)) {
        return {};
    }
    std::span<MemoryType const> const tail = std::views::drop_while(
        mem_types, [&](MemoryType const& mt) { return mt != from; }
    );
    auto const to_it = std::ranges::find(tail, to);
    // Build the span from iterators to avoid a signed to unsigned size conversion.
    return to_it == tail.end() ? std::span<MemoryType const>{}
                               : std::span<MemoryType const>{tail.begin(), to_it + 1};
}

/// @brief Memory types that expose an addressable host or device pointer.
/// @note This is an ordered subset of `MEMORY_TYPES` based on decreasing preference.
constexpr std::span<MemoryType const> ADDRESSABLE_MEMORY_TYPES =
    preference_between(MemoryType::DEVICE, MemoryType::HOST);

/**
 * @brief Get the memory types with preference lower than or equal to @p mem_type.
 *
 * The returned span is a suffix of @p mem_types, which must list memory types in
 * decreasing order of preference.
 *
 * @param mem_type The memory type used as the starting point.
 * @param mem_types The memory types to select from, `ADDRESSABLE_MEMORY_TYPES` by
 * default.
 * @return A span of memory types whose preference is lower than or equal to
 * the given type, empty if @p mem_type is not in @p mem_types.
 */
constexpr std::span<MemoryType const> preference_leq(
    MemoryType mem_type, std::span<MemoryType const> mem_types = ADDRESSABLE_MEMORY_TYPES
) noexcept {
    return mem_types.empty() ? mem_types
                             : preference_between(mem_type, mem_types.back(), mem_types);
}

/**
 * @brief Get the memory types that data in @p mem_type may be spilled to.
 *
 * The targets are the memory types with a lower preference than @p mem_type, in
 * decreasing order of preference. A target may be unavailable at runtime, e.g. when no
 * pinned memory resource or disk resource is configured.
 *
 * @param mem_type The memory type to spill from.
 * @param mem_types The memory types to select from, in decreasing order of preference,
 * `MEMORY_TYPES` by default.
 * @return A span of spill target memory types, empty if @p mem_type cannot be spilled.
 */
constexpr std::span<MemoryType const> preference_lt(
    MemoryType mem_type, std::span<MemoryType const> mem_types = MEMORY_TYPES
) noexcept {
    auto const leq = preference_leq(mem_type, mem_types);
    return leq.empty() ? leq : leq.subspan(1);  // Drop `mem_type` itself.
}

namespace detail::memory_type_tests {
using enum MemoryType;

// `std::ranges::equal` cannot deduce a braced list, so wrap it.
constexpr bool equal(
    std::span<MemoryType const> actual, std::initializer_list<MemoryType> expected
) {
    return std::ranges::equal(actual, expected);
}

// preference_between
static_assert(
    equal(preference_between(PINNED_HOST, HOST, MEMORY_TYPES), {PINNED_HOST, HOST})
);
static_assert(equal(preference_between(HOST, HOST, MEMORY_TYPES), {HOST}));
// `to` > `from` --> empty
static_assert(preference_between(HOST, DEVICE, MEMORY_TYPES).empty());
static_assert(preference_between(HOST, HOST, std::span<MemoryType>{}).empty());
// an unknown `from` or `to` should return an empty view, even if `mem_types` holds it
constexpr auto UNKNOWN = static_cast<MemoryType>(-1);
constexpr std::array with_unknown{DEVICE, UNKNOWN, HOST};
static_assert(preference_between(UNKNOWN, HOST, with_unknown).empty());
static_assert(preference_between(UNKNOWN, HOST, MEMORY_TYPES).empty());
static_assert(preference_between(HOST, UNKNOWN, MEMORY_TYPES).empty());
static_assert(preference_between(DEVICE, UNKNOWN, with_unknown).empty());

// preference_leq
static_assert(std::ranges::equal(preference_leq(DEVICE), ADDRESSABLE_MEMORY_TYPES));
static_assert(equal(preference_leq(HOST), {HOST}));
static_assert(preference_leq(DISK).empty());
// unknown memory type should return an empty view
static_assert(preference_leq(static_cast<MemoryType>(-1)).empty());
static_assert(equal(preference_leq(HOST, MEMORY_TYPES), {HOST, DISK}));

// preference_lt
static_assert(equal(preference_lt(DEVICE), {PINNED_HOST, HOST, DISK}));
static_assert(equal(preference_lt(PINNED_HOST), {HOST, DISK}));
static_assert(equal(preference_lt(HOST), {DISK}));
static_assert(preference_lt(DISK).empty());
static_assert(preference_lt(static_cast<MemoryType>(-1)).empty());
static_assert(
    equal(preference_lt(DEVICE, ADDRESSABLE_MEMORY_TYPES), {PINNED_HOST, HOST})
);
static_assert(equal(preference_lt(PINNED_HOST, ADDRESSABLE_MEMORY_TYPES), {HOST}));
static_assert(preference_lt(HOST, ADDRESSABLE_MEMORY_TYPES).empty());
static_assert(preference_lt(DISK, ADDRESSABLE_MEMORY_TYPES).empty());
}  // namespace detail::memory_type_tests

/**
 * @brief A value of type @p T for every memory type, indexed by `MemoryType`.
 *
 * @tparam T The value type.
 */
template <typename T>
struct PerMemoryType {
    std::array<T, MEMORY_TYPES.size()> values{};  ///< Values for each memory type

    /**
     * @brief Access the value of a memory type.
     *
     * @param mem_type The memory type.
     * @return Reference to the value of @p mem_type.
     */
    [[nodiscard]] constexpr T& operator[](MemoryType mem_type) noexcept {
        return values[static_cast<std::size_t>(mem_type)];
    }

    /// @copydoc operator[](MemoryType)
    [[nodiscard]] constexpr T const& operator[](MemoryType mem_type) const noexcept {
        return values[static_cast<std::size_t>(mem_type)];
    }
};

/**
 * @brief Get the name of a MemoryType.
 *
 * @param mem_type The memory type.
 * @return The memory type name.
 */
constexpr char const* to_string(MemoryType mem_type) {
    return MEMORY_TYPE_NAMES[static_cast<std::size_t>(mem_type)];
}

/**
 * @brief Overload to write type name to the output stream.
 *
 * @param os The output stream.
 * @param mem_type The memory type to write name of to the output stream.
 * @return The output stream.
 */
std::ostream& operator<<(std::ostream& os, MemoryType mem_type);

/**
 * @brief Overload to read a MemoryType value from an input stream.
 *
 * Parsing is case-insensitive. Supported values are: "DEVICE", "PINNED_HOST",
 * "PINNED", "PINNED-HOST", and "HOST".
 *
 * If token extraction from the stream fails, the stream state is preserved.
 * If extraction succeeds but the token does not represent a valid MemoryType,
 * the stream failbit is set.
 *
 * @param is The input stream.
 * @param out The memory type read from the input stream.
 * @return The input stream.
 */
std::istream& operator>>(std::istream& is, MemoryType& out);

}  // namespace rapidsmpf
