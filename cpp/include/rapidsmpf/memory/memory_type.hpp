/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <array>

namespace rapidsmpf {

/// @brief Enum representing the type of memory.
enum class MemoryType : int {
    DEVICE = 0,  ///< Device memory
    HOST = 1  ///< Host memory
};

/// @brief All memory types sorted in decreasing order of preference.
constexpr std::array<MemoryType, 2> MEMORY_TYPES{{MemoryType::DEVICE, MemoryType::HOST}};

/**
 * @brief Memory types that reside in RAM, sorted in decreasing order of preference.
 *
 * Resident memory types include host and device memory, but exclude remote memory
 * and memory stored on disk. These are the memory kinds typically supported by
 * CUDA-aware libraries such as MPI and UCXX.
 */
constexpr std::array<MemoryType, 2> MEMORY_RESIDENT_TYPES{
    {MemoryType::DEVICE, MemoryType::HOST}
};

}  // namespace rapidsmpf
