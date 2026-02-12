/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <string>

#include <rapidsmpf/bootstrap/types.hpp>

namespace rapidsmpf::bootstrap {

/**
 * @brief Backend types for process coordination and bootstrapping.
 */
enum class BackendType {
    /**
     * @brief Automatically detect the best backend based on environment.
     *
     * Detection order:
     * 1. File-based (if RAPIDSMPF_COORD_DIR set by rrun)
     * 2. Slurm/PMIx (if SLURM environment detected)
     * 3. File-based (default fallback)
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

    /**
     * @brief Slurm-based coordination using PMIx.
     *
     * Uses PMIx (Process Management Interface for Exascale) for scalable process
     * coordination without requiring a shared filesystem. Designed for Slurm clusters
     * and supports multi-node deployments.
     *
     * Run with: `srun --mpi=pmix -n <nranks> ./program`
     *
     * Environment variables (automatically set by Slurm):
     * - PMIX_NAMESPACE: PMIx namespace identifier
     * - SLURM_PROCID: Process rank
     * - SLURM_NPROCS/SLURM_NTASKS: Total number of processes
     */
    SLURM,
};

namespace detail {

/**
 * @brief Abstract interface for bootstrap coordination backends.
 *
 * This interface defines the common operations that all backend implementations
 * must support. Backend instances are stored in Context and reused across
 * multiple operations to preserve state.
 */
class Backend {
  public:
    virtual ~Backend() = default;

    /**
     * @brief Store a key-value pair.
     *
     * The key-value pair is committed immediately and made visible to other
     * ranks after a collective `sync()`.
     *
     * @param key Key name.
     * @param value Value to store.
     */
    virtual void put(std::string const& key, std::string const& value) = 0;

    /**
     * @brief Retrieve a value, blocking until available or timeout occurs.
     *
     * @param key Key name.
     * @param timeout Timeout duration.
     * @return Value associated with key.
     *
     * @throws std::runtime_error if key not found within timeout.
     */
    virtual std::string get(std::string const& key, Duration timeout) = 0;

    /**
     * @brief Perform a barrier synchronization.
     *
     * All ranks must call this before any rank proceeds.
     */
    virtual void barrier() = 0;

    /**
     * @brief Ensure all previous put() operations are globally visible.
     */
    virtual void sync() = 0;

    /**
     * @brief Broadcast data from rank 0 to all ranks.
     *
     * @param data Data buffer (input on rank 0, output on other ranks).
     * @param size Size in bytes.
     *
     * @throws std::runtime_error if broadcast fails or size mismatch occurs.
     */
    virtual void broadcast(void* data, std::size_t size) = 0;

    // Non-copyable, non-movable (backends manage resources)
    Backend(Backend const&) = delete;
    Backend& operator=(Backend const&) = delete;
    Backend(Backend&&) = delete;
    Backend& operator=(Backend&&) = delete;

  protected:
    Backend() = default;
};

}  // namespace detail
}  // namespace rapidsmpf::bootstrap
