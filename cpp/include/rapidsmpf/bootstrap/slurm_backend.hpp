/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rapidsmpf/config.hpp>

#ifdef RAPIDSMPF_HAVE_SLURM

#include <array>
#include <chrono>
#include <string>

#include <pmix.h>

#include <rapidsmpf/bootstrap/bootstrap.hpp>

namespace rapidsmpf::bootstrap::detail {

/**
 * @brief Slurm-based coordination backend using PMIx.
 *
 * This class implements coordination using PMIx (Process Management Interface
 * for Exascale), which provides scalable process coordination without requiring
 * a shared filesystem. It is designed for Slurm clusters and supports multi-node
 * deployments.
 *
 * Usage:
 * ```bash
 * # Passthrough: multiple (4) tasks per node, one task per GPU, two nodes.
 * srun \
 *     --mpi=pmix \
 *     --nodes=2 \
 *     --ntasks-per-node=4 \
 *     --cpus-per-task=36 \
 *     --gpus-per-task=1 \
 *     --gres=gpu:4 \
 *     rrun ./benchmarks/bench_shuffle -C ucxx
 *
 * # Hybrid mode: one task per node, 4 GPUs per task, two nodes.
 * srun \
 *     --mpi=pmix \
 *     --nodes=2 \
 *     --ntasks-per-node=1 \
 *     --cpus-per-task=144 \
 *     --gpus-per-task=4 \
 *     --gres=gpu:4 \
 *     rrun -n 4 ./benchmarks/bench_shuffle -C ucxx
 * ```
 */
class SlurmBackend {
  public:
    /**
     * @brief Construct a Slurm backend using PMIx.
     *
     * Initializes PMIx and retrieves process information from the runtime.
     *
     * @param ctx Bootstrap context containing rank information.
     *
     * @throws std::runtime_error if PMIx initialization fails.
     */
    explicit SlurmBackend(Context ctx);

    /**
     * @brief Destructor - finalizes PMIx.
     */
    ~SlurmBackend();

    // Non-copyable, non-movable (PMIx state is process-global)
    SlurmBackend(SlurmBackend const&) = delete;
    SlurmBackend& operator=(SlurmBackend const&) = delete;
    SlurmBackend(SlurmBackend&&) = delete;
    SlurmBackend& operator=(SlurmBackend&&) = delete;

    /**
     * @brief Store a key-value pair in the PMIx KVS.
     *
     * The key-value pair is committed immediately and made visible to other
     * ranks via a fence operation.
     *
     * @param key Key name.
     * @param value Value to store.
     *
     * @throws std::runtime_error if PMIx operation fails.
     */
    void put(std::string const& key, std::string const& value);

    /**
     * @brief Retrieve a value from the PMIx KVS.
     *
     * Blocks until the key is available or timeout occurs. Uses polling
     * with exponential backoff.
     *
     * @param key Key name.
     * @param timeout Timeout duration.
     * @return Value associated with key.
     *
     * @throws std::runtime_error if key not found within timeout.
     */
    std::string get(std::string const& key, Duration timeout);

    /**
     * @brief Perform a barrier synchronization using PMIx_Fence.
     *
     * All ranks must call this before any rank proceeds. The fence also
     * ensures all committed key-value pairs are visible to all ranks.
     *
     * @throws std::runtime_error if PMIx_Fence fails.
     */
    void barrier();

    /**
     * @brief Ensure all previous put() operations are globally visible.
     *
     * For Slurm/PMIx backend, this executes PMIx_Fence to make all committed
     * key-value pairs visible across all nodes. This is required because
     * PMIx_Put + PMIx_Commit only makes data locally visible; PMIx_Fence
     * performs the global synchronization and data exchange.
     *
     * @throws std::runtime_error if PMIx_Fence fails.
     */
    void sync();

    /**
     * @brief Broadcast data from root to all ranks.
     *
     * Root rank publishes data via put(), then all ranks synchronize
     * and non-root ranks retrieve the data via get().
     *
     * @param data Data buffer (input on root, output on others).
     * @param size Size in bytes.
     * @param root Root rank.
     *
     * @throws std::runtime_error if broadcast fails or size mismatch occurs.
     */
    void broadcast(void* data, std::size_t size, Rank root);

  private:
    Context ctx_;
    std::size_t barrier_count_{0};
    bool pmix_initialized_{false};
    pmix_proc_t proc_{};  ///< PMIx process identifier
    std::array<char, PMIX_MAX_NSLEN + 1> nspace_{};  ///< PMIx namespace (job identifier)

    /**
     * @brief Commit local key-value pairs to make them visible.
     *
     * Must be called after put() operations. The subsequent fence()
     * or barrier() will make the data globally visible.
     *
     * @throws std::runtime_error if PMIx_Commit fails.
     */
    void commit();
};

}  // namespace rapidsmpf::bootstrap::detail

#endif  // RAPIDSMPF_HAVE_SLURM
