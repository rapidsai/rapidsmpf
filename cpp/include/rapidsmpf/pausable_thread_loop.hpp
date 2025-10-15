/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <thread>

#include <rapidsmpf/utils.hpp>

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
     * between executions of the task.
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
     * @brief Resumes execution of the thread after being paused.
     *
     * Calling resume on an already running loop is a no-op and is allowed.
     *
     * @return True if the state is changed to Running, false otherwise.
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
    };

    // State transition matrix:
    // | cur_state |         |          | nxt_state |          |         || Ops     |
    // |           |---------|----------|-----------|----------|---------||         |
    // |           | STOPPED | STOPPING | PAUSED    | PAUSING  | RUNNING ||---------|
    // |-----------|---------|----------|-----------|----------|---------|| e_loop  |
    // | STOPPED   | no_op   | X        | X         | X        | X       || pause   |
    // | STOPPING  | e_loop  | no_op    | X         | X        | X       || resume  |
    // | PAUSED    | X       | stop     | no_op     | X        | resume  || stop    |
    // | PAUSING   | e_loop  | stop     | e_loop    | no_op    | resume  || no_op   |
    // | RUNNING   | X       | stop     | X         | pause    | no_op   || X       |
    //
    // pause():
    //  RUNNING -> PAUSING
    // stop():
    //  RUNNING -> STOPPING | PAUSING -> STOPPING | PAUSED -> STOPPING
    // resume():
    //  PAUSING -> RUNNING | PAUSED -> RUNNING
    // automatic transition via event loop:
    //  PAUSING -> PAUSED
    //  STOPPING -> STOPPED

    std::thread thread_;
    std::atomic<State> state_{State::Paused};
};

}  // namespace rapidsmpf::detail
