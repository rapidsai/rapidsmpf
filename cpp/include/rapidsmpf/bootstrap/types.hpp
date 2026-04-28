/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
 * @brief Maximum allowed length (in bytes) for a KV coordination key.
 *
 * Enforced uniformly by all backend implementations:
 * - FileBackend: matches the POSIX NAME_MAX filename limit (255 bytes).
 * - SocketBackend: protocol field width (`%255s`) matches this value.
 *
 * Keys must also be valid as POSIX filename components: no whitespace,
 * path separators (`/`, `\`), path traversal sequences (e.g. `..`), or null bytes.
 */
inline constexpr std::size_t max_key_size = 255;

}  // namespace rapidsmpf::bootstrap
