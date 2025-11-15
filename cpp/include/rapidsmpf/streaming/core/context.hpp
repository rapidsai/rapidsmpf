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
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/queue.hpp>

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
     * @param executor Unique pointer to a coroutine thread pool.
     * @param br Shared pointer to a buffer resource.
     * @param statistics Shared pointer to a statistics collector.
     */
    Context(
        config::Options options,
        std::shared_ptr<Communicator> comm,
        std::shared_ptr<ProgressThread> progress_thread,
        std::unique_ptr<coro::thread_pool> executor,
        std::shared_ptr<BufferResource> br,
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
        std::shared_ptr<BufferResource> br,
        std::shared_ptr<Statistics> statistics = Statistics::disabled()
    );

    ~Context() noexcept;

    /**
     * @brief Returns the configuration options.
     *
     * @return The Options instance.
     */
    [[nodiscard]] config::Options get_options() const noexcept;

    /**
     * @brief Returns the communicator.
     *
     * @return Shared pointer to the communicator.
     */
    [[nodiscard]] std::shared_ptr<Communicator> comm() const noexcept;

    /**
     * @brief Returns the progress thread.
     *
     * @return Shared pointer to the progress thread.
     */
    [[nodiscard]] std::shared_ptr<ProgressThread> progress_thread() const noexcept;

    /**
     * @brief Returns the coroutine thread pool.
     *
     * @return Reference to unique pointer to the thread pool.
     */
    [[nodiscard]] std::unique_ptr<coro::thread_pool>& executor() noexcept;

    /**
     * @brief Returns the buffer resource.
     *
     * @return Raw pointer to the buffer resource.
     */
    [[nodiscard]] BufferResource* br() const noexcept;

    /**
     * @brief Returns the statistics collector.
     *
     * @return Shared pointer to the statistics instance.
     */
    [[nodiscard]] std::shared_ptr<Statistics> statistics() const noexcept;

    /**
     * @brief Create a new channel associated with this context.
     *
     * @return A shared pointer to the newly created channel.
     */
    [[nodiscard]] std::shared_ptr<Channel> create_channel() const noexcept;

    /**
     * @brief Returns the spillable messages collection.
     *
     * @return A shared pointer to the collection.
     */
    [[nodiscard]] std::shared_ptr<SpillableMessages> spillable_messages() const {
        return spillable_messages_;
    }

    /**
     * @brief Create a new bounded queue associated with this context.
     *
     * @param buffer_size Maximum size of the queue.
     *
     * @return A shared pointer to the newly created bounded queue.
     */
    [[nodiscard]] std::shared_ptr<BoundedQueue> create_bounded_queue(
        std::size_t buffer_size
    ) const noexcept;

  private:
    config::Options options_;
    std::shared_ptr<Communicator> comm_;
    std::shared_ptr<ProgressThread> progress_thread_;
    std::unique_ptr<coro::thread_pool> executor_;
    std::shared_ptr<BufferResource> br_;
    std::shared_ptr<Statistics> statistics_;
    std::shared_ptr<SpillableMessages> spillable_messages_;
    SpillManager::SpillFunctionID spill_function_id_{};
};

}  // namespace rapidsmpf::streaming
