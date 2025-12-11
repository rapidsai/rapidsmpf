/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <array>
#include <ostream>

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
