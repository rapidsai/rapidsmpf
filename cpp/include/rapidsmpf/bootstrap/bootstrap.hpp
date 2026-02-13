/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include <rapidsmpf/bootstrap/backend.hpp>
#include <rapidsmpf/bootstrap/types.hpp>

namespace rapidsmpf::bootstrap {

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

    /** @brief Backend type used for coordination. */
    BackendType type;

    /** @brief Coordination directory (for FILE backend). */
    std::optional<std::string> coord_dir;

    /** @brief Backend implementation (internal, do not access directly). */
    std::shared_ptr<detail::Backend> backend;
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
 * @param type Backend type to use (default: AUTO for auto-detection).
 * @return Context object containing rank and coordination information.
 * @throws std::runtime_error if environment is not properly configured.
 *
 * @code
 * auto ctx = rapidsmpf::bootstrap::init();
 * std::cout << "I am rank " << ctx.rank << " of " << ctx.nranks << std::endl;
 * @endcode
 */
Context init(BackendType type = BackendType::AUTO);

/**
 * @brief Perform a barrier synchronization across all ranks.
 *
 * This ensures all ranks reach this point before any rank proceeds.
 *
 * @param ctx Bootstrap context.
 */
void barrier(Context const& ctx);

/**
 * @brief Ensure all previous put() operations are globally visible.
 *
 * Different backends have different visibility semantics for put() operations:
 * - Slurm/PMIx: Requires explicit fence (PMIx_Fence) to make data visible across nodes.
 * - FILE: put() operations are immediately visible via atomic filesystem operations.
 *
 * This function abstracts these differences. Call sync() after put() operations
 * to ensure data is visible to other ranks before they attempt get().
 *
 * @param ctx Bootstrap context.
 */
void sync(Context const& ctx);

/**
 * @brief Store a key-value pair in the coordination backend (rank 0 only).
 *
 * Only rank 0 should call this function. The key-value pair is made visible
 * to all ranks after a `sync()` call. Use this for custom coordination such
 * as UCXX address exchange.
 *
 * @param ctx Bootstrap context.
 * @param key Key name.
 * @param value Value to store.
 *
 * @throws std::runtime_error if called by non-zero rank.
 */
void put(Context const& ctx, std::string const& key, std::string const& value);

/**
 * @brief Retrieve a value from the coordination backend.
 *
 * Any rank (including rank 0) can call this function to retrieve values
 * published by rank 0. This function blocks until the key is available
 * or timeout occurs.
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
