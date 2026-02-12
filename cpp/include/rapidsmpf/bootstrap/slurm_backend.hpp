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

#include <rapidsmpf/bootstrap/backend.hpp>
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
 * # Multiple (4) tasks per node, one task per GPU, two nodes.
 * srun \
 *     --mpi=pmix \
 *     --nodes=2 \
 *     --ntasks-per-node=4 \
 *     --cpus-per-task=36 \
 *     --gpus-per-task=1 \
 *     --gres=gpu:4 \
 *     rrun ./benchmarks/bench_shuffle -C ucxx
 * ```
 */
class SlurmBackend : public Backend {
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
     * @brief Destructor that finalizes PMIx.
     */
    ~SlurmBackend() override;

    // Non-copyable, non-movable (PMIx state is process-global)
    SlurmBackend(SlurmBackend const&) = delete;
    SlurmBackend& operator=(SlurmBackend const&) = delete;
    SlurmBackend(SlurmBackend&&) = delete;
    SlurmBackend& operator=(SlurmBackend&&) = delete;

    /**
     * @copydoc Backend::put()
     *
     * @throws std::runtime_error if PMIx operation fails.
     */
    void put(std::string const& key, std::string const& value) override;

    /**
     * @copydoc Backend::get()
     */
    std::string get(std::string const& key, Duration timeout) override;

    /**
     * @copydoc Backend::barrier()
     *
     * @throws std::runtime_error if PMIx_Fence fails.
     */
    void barrier() override;

    /**
     * @copydoc Backend::sync()
     *
     * @throws std::runtime_error if PMIx_Fence fails.
     */
    void sync() override;

  private:
    Context ctx_;
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
