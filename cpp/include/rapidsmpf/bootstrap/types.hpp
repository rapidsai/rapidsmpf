/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <cstdint>

namespace rapidsmpf::bootstrap {

/// @brief Type alias for communicator::Rank
using Rank = std::int32_t;

/// @brief Type alias for Duration type
using Duration = std::chrono::duration<double>;

}  // namespace rapidsmpf::bootstrap
