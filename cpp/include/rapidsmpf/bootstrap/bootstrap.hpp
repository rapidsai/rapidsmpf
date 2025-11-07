/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include <rapidsmpf/config.hpp>
#include <rapidsmpf/utils.hpp>

namespace rapidsmpf::bootstrap {

/// @brief Type alias for communicator::Rank
using Rank = std::int32_t;

/// @brief Type alias for duration type
using rapidsmpf::Duration;

/**
 * @brief Backend types for process coordination and bootstrapping.
 */
enum class Backend {
    /**
     * @brief Automatically detect the best backend based on environment.
     *
     * Detection order:
     * 1. File-based (default fallback)
     */
    AUTO,

    /**
     * @brief File-based coordination using a shared directory.
     *
     * Uses filesystem for rank coordination and address exchange.  Works on single-node
     * and multi-node with shared storage (e.g., NFS) via SSH. Requires RAPIDSMPF_RANK,
     * RAPIDSMPF_NRANKS, RAPIDSMPF_COORD_DIR environment variables.
     */
    FILE,
};

/**
 * @brief Context information for the current process/rank.
 *
 * This structure contains the rank assignment and total rank count,
 * along with additional metadata about the execution environment.
 */
struct Context {
    /** @brief This process's rank (0-indexed). */
    Rank rank;

    /** @brief Total number of ranks in the job. */
    Rank nranks;

    /** @brief Backend used for coordination. */
    Backend backend;

    /** @brief Coordination directory (for FILE backend). */
    std::optional<std::string> coord_dir;
};

/**
 * @brief Initialize the bootstrap context from environment variables.
 *
 * This function reads environment variables to determine rank, nranks, and
 * backend configuration. It should be called early in the application lifecycle.
 *
 * Environment variables checked (in order of precedence):
 * - RAPIDSMPF_RANK: Explicitly set rank
 * - RAPIDSMPF_NRANKS: Explicitly set total rank count
 * - RAPIDSMPF_COORD_DIR: File-based coordination directory
 *
 * @param backend Backend to use (default: AUTO for auto-detection).
 * @return Context object containing rank and coordination information.
 * @throws std::runtime_error if environment is not properly configured.
 *
 * @code
 * auto ctx = rapidsmpf::bootstrap::init();
 * std::cout << "I am rank " << ctx.rank << " of " << ctx.nranks << std::endl;
 * @endcode
 */
Context init(Backend backend = Backend::AUTO);

/**
 * @brief Broadcast data from root rank to all other ranks.
 *
 * This is a helper function for broadcasting small amounts of data during
 * bootstrapping. It uses the underlying backend's coordination mechanism.
 *
 * @param ctx Bootstrap context.
 * @param data Data buffer to broadcast (both input on root, output on others).
 * @param size Size of data in bytes.
 * @param root Root rank performing the broadcast (default: 0).
 */
void broadcast(Context const& ctx, void* data, std::size_t size, Rank root = 0);

/**
 * @brief Perform a barrier synchronization across all ranks.
 *
 * This ensures all ranks reach this point before any rank proceeds.
 *
 * @param ctx Bootstrap context.
 */
void barrier(Context const& ctx);

/**
 * @brief Store a key-value pair in the coordination backend.
 *
 * This is useful for custom coordination beyond UCXX address exchange.
 *
 * @param ctx Bootstrap context.
 * @param key Key name.
 * @param value Value to store.
 */
void put(Context const& ctx, std::string const& key, std::string const& value);

/**
 * @brief Retrieve a value from the coordination backend.
 *
 * This function blocks until the key is available or timeout occurs.
 *
 * @param ctx Bootstrap context.
 * @param key Key name to retrieve.
 * @param timeout Timeout duration.
 * @return Value associated with the key.
 * @throws std::runtime_error if key not found within timeout.
 */
std::string get(
    Context const& ctx,
    std::string const& key,
    Duration timeout = std::chrono::seconds{30}
);

}  // namespace rapidsmpf::bootstrap
