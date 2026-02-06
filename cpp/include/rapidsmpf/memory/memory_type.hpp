/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <algorithm>
#include <array>
#include <iostream>
#include <ranges>
#include <span>

#include <rapidsmpf/utils/misc.hpp>

namespace rapidsmpf {

/// @brief Enum representing the type of memory sorted in decreasing order of preference.
enum class MemoryType : int {
    DEVICE = 0,  ///< Device memory
    PINNED_HOST = 1,  ///< Pinned host memory
    HOST = 2  ///< Host memory
};

/// @brief All memory types sorted in decreasing order of preference.
constexpr std::array<MemoryType, 3> MEMORY_TYPES{
    {MemoryType::DEVICE, MemoryType::PINNED_HOST, MemoryType::HOST}
};

/// @brief Memory type names sorted to match `MemoryType` and `MEMORY_TYPES`.
constexpr std::array<char const*, MEMORY_TYPES.size()> MEMORY_TYPE_NAMES{
    {"DEVICE", "PINNED_HOST", "HOST"}
};

/**
 * @brief Memory types that are valid spill destinations in decreasing order of
 * preference.
 *
 * This array defines the preferred targets for spilling when device memory is
 * insufficient. The ordering reflects the policy of spilling in RapidsMPF, where
 * earlier entries are considered more desirable spill destinations.
 */
constexpr std::array<MemoryType, 2> SPILL_TARGET_MEMORY_TYPES{
    {MemoryType::PINNED_HOST, MemoryType::HOST}
};

/**
 * @brief Get the memory types with preference lower than or equal to @p mem_type.
 *
 * The returned span reflects the predefined ordering used in \c MEMORY_TYPES,
 * which lists memory types in decreasing order of preference.
 *
 * @param mem_type The memory type used as the starting point.
 * @return A span of memory types whose preference is lower than or equal to
 * the given type.
 */
constexpr std::span<MemoryType const> leq_memory_types(MemoryType mem_type) noexcept {
    return std::views::drop_while(MEMORY_TYPES, [&](MemoryType const& mt) {
        return mt != mem_type;
    });
}

static_assert(std::ranges::equal(leq_memory_types(MemoryType::DEVICE), MEMORY_TYPES));
static_assert(std::ranges::equal(
    leq_memory_types(MemoryType::HOST), std::ranges::single_view{MemoryType::HOST}
));
// unknown memory type should return an empty view
static_assert(std::ranges::equal(
    leq_memory_types(static_cast<MemoryType>(-1)), std::ranges::empty_view<MemoryType>{}
));

/**
 * @brief Get the memory types that are device accessible.
 *
 * @return A span of memory types that are device accessible.
 */
constexpr std::span<MemoryType const> device_accessible_memory_types() noexcept {
    return std::span{MEMORY_TYPES}.first<2>();
}

static_assert(std::ranges::equal(
    device_accessible_memory_types(),
    std::array{MemoryType::DEVICE, MemoryType::PINNED_HOST}
));

/**
 * @brief Check if a memory type is device accessible.
 *
 * @param mem_type The memory type to check.
 * @return true if the memory type is device accessible, false otherwise.
 */
constexpr bool is_device_accessible(MemoryType mem_type) noexcept {
    return contains(device_accessible_memory_types(), mem_type);
}

/**
 * @brief Get the memory types that are host accessible.
 *
 * @return A span of memory types that are host accessible.
 */
constexpr std::span<MemoryType const> host_accessible_memory_types() {
    return std::span{MEMORY_TYPES}.last<2>();
}

static_assert(std::ranges::equal(
    host_accessible_memory_types(), std::array{MemoryType::PINNED_HOST, MemoryType::HOST}
));

/**
 * @brief Check if a memory type is host accessible.
 *
 * @param mem_type The memory type to check.
 * @return true if the memory type is host accessible, false otherwise.
 */
constexpr bool is_host_accessible(MemoryType mem_type) noexcept {
    return contains(host_accessible_memory_types(), mem_type);
}

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
