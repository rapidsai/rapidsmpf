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

/// @brief Memory types that expose an addressable host or device pointer.
/// @note This is an ordered subset of `MEMORY_TYPES` based on decreasing preference.
constexpr std::array ADDRESSABLE_MEMORY_TYPES{
    MemoryType::DEVICE, MemoryType::PINNED_HOST, MemoryType::HOST
};

/// @brief Memory type names sorted to match `MemoryType` and `MEMORY_TYPES`.
constexpr std::array<char const*, MEMORY_TYPES.size()> MEMORY_TYPE_NAMES{
    {"DEVICE", "PINNED_HOST", "HOST", "DISK"}
};

/**
 * @brief A value of type @p T for every memory type, indexed by `MemoryType`.
 *
 * @tparam T The value type.
 */
template <typename T>
struct PerMemoryType {
    std::array<T, MEMORY_TYPES.size()> values{};

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

// Data waiting to be sent may be spilled, so the spill targets must be addressable.
static_assert(std::ranges::all_of(SPILL_TARGET_MEMORY_TYPES, [](MemoryType mt) {
    return std::ranges::find(ADDRESSABLE_MEMORY_TYPES, mt)
           != ADDRESSABLE_MEMORY_TYPES.end();
}));

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
constexpr std::span<MemoryType const> leq_memory_types(
    MemoryType mem_type, std::span<MemoryType const> mem_types = ADDRESSABLE_MEMORY_TYPES
) noexcept {
    return std::views::drop_while(mem_types, [&](MemoryType const& mt) {
        return mt != mem_type;
    });
}

static_assert(
    std::ranges::equal(leq_memory_types(MemoryType::DEVICE), ADDRESSABLE_MEMORY_TYPES)
);
static_assert(std::ranges::equal(
    leq_memory_types(MemoryType::HOST), std::ranges::single_view{MemoryType::HOST}
));
static_assert(std::ranges::equal(
    leq_memory_types(MemoryType::DISK), std::ranges::empty_view<MemoryType>{}
));
// unknown memory type should return an empty view
static_assert(std::ranges::equal(
    leq_memory_types(static_cast<MemoryType>(-1)), std::ranges::empty_view<MemoryType>{}
));
static_assert(std::ranges::equal(
    leq_memory_types(MemoryType::HOST, MEMORY_TYPES),
    std::array{MemoryType::HOST, MemoryType::DISK}
));

/**
 * @brief Get the memory types that data in @p mem_type may be spilled to.
 *
 * The targets are the memory types with a lower preference than @p mem_type, in
 * decreasing order of preference. A target may be unavailable at runtime, e.g. when no
 * pinned memory resource or disk resource is configured.
 *
 * @param mem_type The memory type to spill from.
 * @return A span of spill target memory types, empty if @p mem_type cannot be spilled.
 */
constexpr std::span<MemoryType const> spill_target_memory_types(
    MemoryType mem_type
) noexcept {
    auto const leq = leq_memory_types(mem_type, MEMORY_TYPES);
    // Drop `mem_type` itself.
    return leq.empty() ? leq : leq.subspan(1);
}

static_assert(std::ranges::equal(
    spill_target_memory_types(MemoryType::DEVICE),
    std::array{MemoryType::PINNED_HOST, MemoryType::HOST, MemoryType::DISK}
));
static_assert(std::ranges::equal(
    spill_target_memory_types(MemoryType::PINNED_HOST),
    std::array{MemoryType::HOST, MemoryType::DISK}
));
static_assert(std::ranges::equal(
    spill_target_memory_types(MemoryType::HOST), std::array{MemoryType::DISK}
));
static_assert(spill_target_memory_types(MemoryType::DISK).empty());
static_assert(spill_target_memory_types(static_cast<MemoryType>(-1)).empty());

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
