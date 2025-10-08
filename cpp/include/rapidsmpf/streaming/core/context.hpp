/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/config.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/progress_thread.hpp>
#include <rapidsmpf/statistics.hpp>

#include <coro/coro.hpp>

namespace rapidsmpf::streaming {

/**
 * @brief Context for nodes (coroutines) in rapidsmpf.
 */
class Context {
  public:
    /**
     * @brief Full constructor for the Context.
     *
     * All provided pointers must be non-null.
     *
     * @param options Configuration options.
     * @param comm Shared pointer to a communicator.
     * @param progress_thread Shared pointer to a progress thread.
     * @param executor Shared pointer to a coroutine thread pool.
     * @param br Shared pointer to a buffer resource.
     * @param statistics Shared pointer to a statistics collector.
     */
    Context(
        config::Options options,
        std::shared_ptr<Communicator> comm,
        std::shared_ptr<ProgressThread> progress_thread,
        std::shared_ptr<coro::thread_pool> executor,
        BufferResource* br,
        std::shared_ptr<Statistics> statistics
    );

    /**
     * @brief Convenience constructor with minimal configuration.
     *
     * Creates a default ProgressThread and coroutine thread pool.
     *
     * @param options Configuration options.
     * @param comm Shared pointer to a communicator.
     * @param br Buffer resource used to reserve host memory and perform the move.
     * @param statistics The statistics instance to use (disabled by default).
     */
    Context(
        config::Options options,
        std::shared_ptr<Communicator> comm,
        BufferResource* br,
        std::shared_ptr<Statistics> statistics = Statistics::disabled()
    );

    /**
     * @brief Returns the configuration options.
     *
     * @return The Options instance.
     */
    config::Options get_options();

    /**
     * @brief Returns the communicator.
     *
     * @return Shared pointer to the communicator.
     */
    std::shared_ptr<Communicator> comm();

    /**
     * @brief Returns the progress thread.
     *
     * @return Shared pointer to the progress thread.
     */
    std::shared_ptr<ProgressThread> progress_thread();

    /**
     * @brief Returns the coroutine thread pool.
     *
     * @return Shared pointer to the thread pool.
     */
    std::shared_ptr<coro::thread_pool> executor();

    /**
     * @brief Returns the buffer resource.
     *
     * @return Raw pointer to the buffer resource.
     */
    BufferResource* br();

    /**
     * @brief Returns the statistics collector.
     *
     * @return Shared pointer to the statistics instance.
     */
    std::shared_ptr<Statistics> statistics();

  private:
    config::Options options_;
    std::shared_ptr<Communicator> comm_;
    std::shared_ptr<ProgressThread> progress_thread_;
    std::shared_ptr<coro::thread_pool> executor_;
    BufferResource* br_;
    std::shared_ptr<Statistics> statistics_;
};

}  // namespace rapidsmpf::streaming
