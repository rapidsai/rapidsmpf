/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <array>
#include <ostream>
#include <ranges>
#include <span>

namespace rapidsmpf {

/// @brief Enum representing the type of memory.
enum class MemoryType : int {
    DEVICE = 0,  ///< Device memory
    HOST = 1  ///< Host memory
};

/// @brief All memory types sorted in decreasing order of preference.
constexpr std::array<MemoryType, 2> MEMORY_TYPES{{MemoryType::DEVICE, MemoryType::HOST}};

/// @brief Memory type names sorted to match `MEMORY_TYPES`.
constexpr std::array<char const*, MEMORY_TYPES.size()> MEMORY_TYPE_NAMES{
    {"DEVICE", "HOST"}
};

/**
 * @brief Get the lower memory types than or equal to the @p mem_type .
 *
 * @param mem_type The memory type.
 * @return A span of the lower memory types than the given memory type.
 */
constexpr std::span<MemoryType const> leq_memory_types(MemoryType mem_type) noexcept {
    return std::span<MemoryType const>{
        MEMORY_TYPES.begin() + static_cast<std::size_t>(mem_type), MEMORY_TYPES.end()
    };
}

static_assert(std::ranges::equal(leq_memory_types(MemoryType::DEVICE), MEMORY_TYPES));
static_assert(std::ranges::equal(
    leq_memory_types(MemoryType::HOST), std::ranges::single_view{MemoryType::HOST}
));

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
inline std::ostream& operator<<(std::ostream& os, MemoryType mem_type) {
    return os << to_string(mem_type);
}

}  // namespace rapidsmpf
