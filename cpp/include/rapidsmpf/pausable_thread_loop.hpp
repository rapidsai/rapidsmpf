/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>

#include <rapidsmpf/utils/misc.hpp>

namespace rapidsmpf::detail {
/**
 * @brief A thread loop that can be paused, resumed, and stopped.
 *
 * This class runs a provided function repeatedly in a separate thread.
 */
class PausableThreadLoop {
  public:
    /**
     * @brief Constructs a thread to run the specified function in a loop.
     *
     * The loop will execute the given function repeatedly in a separate thread.
     * The thread starts in a paused state and will remain paused until the `resume()`
     * method is called.
     *
     * If a sleep duration is provided, the thread will sleep for the specified duration
     * between executions of the task. The sleep is interrupted by `resume()`,
     * `pause_nb()`, `pause()`, and `stop()`.
     *
     * A short sleep is introduced when running under Valgrind to avoid potential issues
     * with thread starvation.
     *
     * @param func The function to execute repeatedly in the thread.
     * @param sleep The duration to sleep between executions of the task. By default, the
     * thread yields execution instead of sleeping.
     */
    PausableThreadLoop(
        std::function<void()> func, Duration sleep = std::chrono::seconds{0}
    );
    ~PausableThreadLoop() noexcept;

    /**
     * @brief Checks if the thread is currently running (not paused or stopped).
     *
     * @note If false, the loop function might still be in the middle of running its
     * last iteration before being paused or stopped.
     *
     * @return True if the thread is running, false if paused or stopped.
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
     * @brief Resumes the thread and requests a prompt iteration of the loop function.
     *
     * If the thread is paused or pausing, it is resumed. If it is sleeping, it starts
     * its next iteration immediately. If it is in the middle of an iteration, the sleep
     * after that iteration is skipped. Multiple calls before the next iteration starts
     * are coalesced into one.
     *
     * @return True if the thread was paused or pausing, false otherwise.
     */
    bool resume() noexcept;

    /**
     * @brief Stops the execution of the thread and joins it.
     *
     * Once stopped, the thread cannot be resumed.
     *
     * @note This function is blocking and will wait on the loop function
     * to finish its current execution.
     *
     * @return True if the thread is stopped by this call, false otherwise (if it was
     * already stopping/stopped).
     */
    bool stop() noexcept;

  private:
    /**
     * @brief The state of the thread loop.
     */
    enum State : std::uint8_t {
        Stopped,  ///< Thread stopped (cannot be resumed)
        Stopping,  ///< Thread is stopping (transitioning to Stopped)
        Paused,  ///< Thread is paused (can be resumed)
        Pausing,  ///< Thread is pausing (transitioning to Paused)
        Running,  ///< Thread is running
        SkipSleep,  ///< Thread is running, and starts its next iteration without sleeping
    };

    // State transition matrix (e_loop: made by the event loop thread, X: impossible):
    // | cur \ nxt  | STOPPED | STOPPING | PAUSED | PAUSING | RUNNING | SKIP_SLEEP |
    // |------------|---------|----------|--------|---------|---------|------------|
    // | STOPPED    | no_op   | X        | X      | X       | X       | X          |
    // | STOPPING   | e_loop  | no_op    | X      | X       | X       | X          |
    // | PAUSED     | X       | stop     | no_op  | X       | X       | resume     |
    // | PAUSING    | e_loop  | stop     | e_loop | no_op   | X       | resume     |
    // | RUNNING    | X       | stop     | X      | pause   | no_op   | resume     |
    // | SKIP_SLEEP | X       | stop     | X      | pause   | e_loop  | no_op      |
    //
    // pause():
    //  RUNNING -> PAUSING | SKIP_SLEEP -> PAUSING
    // stop():
    //  RUNNING -> STOPPING | SKIP_SLEEP -> STOPPING | PAUSING -> STOPPING
    //  | PAUSED -> STOPPING
    // resume():
    //  PAUSED -> SKIP_SLEEP | PAUSING -> SKIP_SLEEP | RUNNING -> SKIP_SLEEP
    // automatic transition via event loop:
    //  PAUSING -> PAUSED
    //  STOPPING -> STOPPED
    //  SKIP_SLEEP -> RUNNING (at the start of an iteration)

    // `state_` is guarded by `mutex_`. Every change except the event loop's
    // SKIP_SLEEP -> RUNNING is followed by `cv_.notify_all()`, since no thread waits on
    // it. The loop function never runs while `mutex_` is held, so
    // callers may hold their own locks (also taken by the loop function) while calling
    // into this class.
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    State state_{State::Paused};
    std::thread thread_;
};

}  // namespace rapidsmpf::detail
