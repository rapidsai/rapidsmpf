/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <exception>
#include <functional>
#include <memory>
#include <thread>

#include <rapidsmpf/utils/misc.hpp>

namespace rapidsmpf::detail {

/// @brief The state machine of a `PausableThreadLoop`, internal to its implementation.
class LoopState;

/**
 * @brief A thread loop that can be paused, resumed, and stopped.
 *
 * This class runs a provided function repeatedly in a separate thread.
 *
 * Exceptions never escape the thread: if the loop function (or the per-thread
 * initialization) throws, the exception is captured, the loop becomes failed or
 * stopped, and the exception is reported to the error handler. The loop is stopped if
 * the failure happens while it is being stopped, in which case the stop completes
 * early, otherwise it enters a terminal failed state. Either way, it never calls the
 * loop function again and cannot be paused or resumed. A failed loop is still stopped
 * (and its thread joined) as usual, by `stop()` or the destructor.
 */
class PausableThreadLoop {
  public:
    /**
     * @brief Handler invoked on the loop thread when the loop function throws.
     *
     * The handler receives the captured exception. It must not throw, the process is
     * terminated if it does. It may call `stop()`, but must not destroy the loop.
     */
    using ErrorHandler = std::function<void(std::exception_ptr)>;

    /**
     * @brief Constructs a thread to run the specified function in a loop.
     *
     * The loop will execute the given function repeatedly in a separate thread.
     * The thread starts in a paused state and will remain paused until the `resume()`
     * method is called.
     *
     * If a sleep duration is provided, the thread will sleep for the specified duration
     * between executions of the task.
     *
     * A short sleep is introduced when running under Valgrind to avoid potential issues
     * with thread starvation.
     *
     * @param func The function to execute repeatedly in the thread.
     * @param sleep The duration to sleep between executions of the task. By default, the
     * thread yields execution instead of sleeping.
     * @param on_error Handler invoked on the loop thread with the exception thrown by
     * `func` (or by the per-thread initialization). By the time it is invoked, the loop
     * has already become failed or stopped (see `error()`). If empty, an exception
     * escaping `func` is fatal and terminates the process with a descriptive message.
     */
    PausableThreadLoop(
        std::function<void()> func,
        Duration sleep = std::chrono::seconds{0},
        ErrorHandler on_error = nullptr
    );

    /**
     * @brief Stops and joins the thread.
     *
     * Must not be called from the loop thread, i.e., from the loop function or the error
     * handler, since a thread cannot join itself. Doing so terminates the process.
     */
    ~PausableThreadLoop() noexcept;

    /**
     * @brief Checks if the thread is currently running (not paused, failed or stopped).
     *
     * @note If false, the loop function might still be in the middle of running its
     * last iteration before being paused or stopped.
     *
     * @return True if the thread is running, false if paused, failed or stopped.
     */
    [[nodiscard]] bool is_running() const noexcept;

    /**
     * @brief Pauses the execution of the thread.
     *
     * The thread will stop executing the loop function until `resume()` is called.
     *
     * @note This function is non-blocking and will let the loop function finish its
     * current execution asynchronously.
     *
     * @warning After calling `pause_nb` the thread is not paused
     * until the state has changed to `State::PAUSED`, so immediately
     * notifying a thread that is waiting on `is_running` will not
     * necessarily wake it.
     */
    void pause_nb() noexcept;

    /**
     * @brief Pauses the execution of the thread.
     *
     * The thread will stop executing the loop function until
     * `resume()` is called.
     *
     * @note Pausing the thread does not interrupt the current iteration.
     */
    void pause() noexcept;

    /**
     * @brief Resumes execution of the thread after being paused.
     *
     * Calling resume on an already running loop is a no-op and is allowed. Resuming a
     * loop that has failed (see `error()`) is a no-op.
     *
     * @return True if the state is changed to Running, false otherwise.
     */
    bool resume() noexcept;

    /**
     * @brief Get the exception that caused the loop to fail, if any.
     *
     * Once this returns an exception, the loop has already failed or stopped: from then
     * on `is_running()` returns false, and pausing or resuming it are no-ops. This holds
     * even if the failure happened while the loop was being stopped, in which case the
     * stop completes early. The error handler may still be running on the loop thread.
     *
     * @return The captured exception, or a null `std::exception_ptr` if the loop
     * function has never thrown.
     */
    [[nodiscard]] std::exception_ptr error() const noexcept;

    /**
     * @brief Stops the execution of the thread and joins it.
     *
     * Once stopped, the thread cannot be resumed.
     *
     * @note This function is blocking and will wait on the loop function
     * to finish its current execution, or for the sleep between iterations to elapse
     * if called during that sleep.
     *
     * @note If called from the loop thread, i.e., from the loop function or the error
     * handler, the thread is not joined: it exits once the current iteration returns,
     * without sleeping, and the destructor joins it.
     *
     * @return True if the thread is stopped by this call, false otherwise (if it was
     * already stopping/stopped).
     */
    bool stop() noexcept;

  private:
    /**
     * @brief Record a failure of the loop function and report it to `on_error_`.
     *
     * Called on the loop thread only.
     *
     * @param error The captured exception.
     */
    void fail(std::exception_ptr error) noexcept;

    ErrorHandler on_error_;  ///< Handler for exceptions thrown by the loop function.
    std::unique_ptr<LoopState> state_;  ///< Created before the thread starts.
    std::thread thread_;
};

}  // namespace rapidsmpf::detail
