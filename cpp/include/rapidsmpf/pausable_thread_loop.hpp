/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
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
    ~PausableThreadLoop();

    /**
     * @brief Checks if the thread is currently running (not paused).
     *
     * @note If false, the loop function might still be in the middle of running its
     * last iteration before being paused.
     *
     * @return True if the thread is running, false if paused.
     */
    [[nodiscard]] bool is_running() const noexcept;

    /**
     * @brief Pauses the execution of the thread.
     *
     * The thread will stop executing the loop function until `resume()` is called.
     *
     * @note This function is non-blocking and will let the loop function finish its
     * current execution asynchronously.
     */
    void pause();

    /**
     * @brief Resumes execution of the thread after being paused.
     *
     * Calling resume on an already running loop is a no-op and is allowed.
     */
    void resume();

    /**
     * @brief Stops the execution of the thread and joins it.
     *
     * Once stopped, the thread cannot be resumed.
     *
     * @note This function is blocking and will wait on the loop function
     * to finish its current execution.
     */
    void stop();

  private:
    std::thread thread_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool active_{true};
    bool paused_{true};
};

}  // namespace rapidsmpf::detail
