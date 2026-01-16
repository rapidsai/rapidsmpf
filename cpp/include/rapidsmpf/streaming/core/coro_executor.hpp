/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <memory>
#include <thread>

#include <rapidsmpf/config.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/statistics.hpp>

#include <coro/coro.hpp>

namespace rapidsmpf::streaming {

/**
 * @brief Executor wrapper around a `coro::thread_pool` used for coroutine execution.
 *
 * The executor lifetime defines the lifetime of the underlying thread pool.
 * The number of threads can be provided explicitly or derived from configuration
 * options.
 *
 * @warning Shutdown of the executor must be initiated from the same thread that
 * created it. Calling `shutdown()` from a different thread results in program
 * termination. Since the destructor implicitly calls `shutdown()`, destroying
 * the executor from a different thread also results in termination unless the
 * executor has already been shut down explicitly.
 *
 * This can be subtle in coroutine-based code, where a scheduled coroutine may
 * unwind its stack on a different thread and trigger destructors. Explicitly
 * calling `shutdown()` on the creator thread allows the destructor to run safely
 * on any thread afterward.
 */
class CoroThreadPoolExecutor {
  public:
    /**
     * @brief Construct an executor with an explicit number of streaming threads.
     *
     * @param num_streaming_threads Number of threads used to execute coroutines.
     * Must be greater than zero.
     * @param statistics Statistics collector associated with the executor. If not
     * provided, statistics collection is disabled. TODO: statistics are not
     * currently collected. In the future, libcoro's thread start and stop callbacks
     * should be used to track coroutine execution statistics.
     */
    CoroThreadPoolExecutor(
        std::uint32_t num_streaming_threads,
        std::shared_ptr<Statistics> statistics = Statistics::disabled()
    );

    /**
     * @brief Construct an executor from configuration options.
     *
     * Reads the `num_streaming_threads` option. If the option is not set,
     * a single streaming thread is used by default.
     *
     * @param options Configuration options used to initialize the executor.
     * @param statistics Statistics collector associated with the executor. If not
     * provided, statistics collection is disabled. TODO: statistics are not
     * currently collected. In the future, libcoro's thread start and stop callbacks
     * should be used to track coroutine execution statistics.
     *
     * @throws std::invalid_argument If `num_streaming_threads` is present but not a
     * positive integer.
     */
    CoroThreadPoolExecutor(
        config::Options options,
        std::shared_ptr<Statistics> statistics = Statistics::disabled()
    );

    ~CoroThreadPoolExecutor() noexcept;

    /// @brief No move and copy constructors and assignment operators.
    CoroThreadPoolExecutor(CoroThreadPoolExecutor&&) = delete;
    CoroThreadPoolExecutor(CoroThreadPoolExecutor const&) = delete;
    CoroThreadPoolExecutor& operator=(CoroThreadPoolExecutor& o) = delete;
    CoroThreadPoolExecutor& operator=(CoroThreadPoolExecutor&& o) = delete;

    /**
     * @brief Shut down the underlying thread pool.
     *
     * This method is idempotent and only performs shutdown once. Subsequent calls
     * have no effect.
     *
     * @warning Shutdown must be initiated from the same thread that constructed
     * the executor. Calling this method from a different thread results in program
     * termination.
     */
    void shutdown() noexcept;

    /**
     * @brief Get the configured number of streaming threads.
     *
     * @return Number of threads in the underlying libcoro thread pool.
     */
    [[nodiscard]] std::uint32_t num_streaming_threads() const noexcept {
        return executor_->thread_count();
    }

    /**
     * @brief Get access to the underlying thread pool to be used with libcoro.
     *
     * @return Reference to the owning `std::unique_ptr` holding the `coro::thread_pool`.
     *
     * @note Ownership of the thread pool remains with the executor.
     */
    [[nodiscard]] std::unique_ptr<coro::thread_pool>& get() noexcept;

    /**
     * @brief Schedule work on the underlying libcoro thread pool.
     *
     * @return A libcoro awaitable as returned by `coro::thread_pool::schedule()`.
     */
    [[nodiscard]] auto schedule() {
        return executor_->schedule();
    }

    /**
     * @brief Schedule a task on the underlying libcoro thread pool.
     *
     * @param task Task to schedule.
     * @return A libcoro awaitable as returned by `coro::thread_pool::schedule(task)`.
     */
    [[nodiscard]] auto schedule(auto task) {
        return executor_->schedule(std::move(task));
    }

    /**
     * @brief Yield execution back to the underlying libcoro thread pool.
     *
     * @return A libcoro awaitable as returned by `coro::thread_pool::yield()`.
     */
    [[nodiscard]] auto yield() {
        return executor_->yield();
    }

    /**
     * @brief Spawn a detached task on the underlying libcoro thread pool.
     *
     * @param task Task to spawn.
     * @return Result as returned by `coro::thread_pool::spawn_detached(task)`.
     */
    auto spawn_detached(auto task) noexcept {
        return executor_->spawn_detached(std::move(task));
    }

    /**
     * @brief Spawn a joinable task on the underlying libcoro thread pool.
     *
     * @param task Task to spawn.
     * @return Result as returned by `coro::thread_pool::spawn_joinable(task)`.
     */
    auto spawn_joinable(auto task) noexcept {
        return executor_->spawn_joinable(std::move(task));
    }

  private:
    std::atomic<bool> is_shutdown_{false};
    std::unique_ptr<coro::thread_pool> executor_;
    std::shared_ptr<Statistics> statistics_;
    std::thread::id creator_thread_id_;
};

}  // namespace rapidsmpf::streaming
