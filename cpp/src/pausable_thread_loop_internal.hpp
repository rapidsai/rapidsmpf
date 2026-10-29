/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <condition_variable>
#include <cstdint>
#include <exception>
#include <mutex>

namespace rapidsmpf::detail {

/**
 * @brief The state machine of a `PausableThreadLoop`, without the thread.
 *
 * Internal to `PausableThreadLoop`, kept out of the public headers but separate from its
 * implementation so that the state machine can be tested on its own. Every operation
 * runs under a single mutex, so concurrent operations behave as if executed in some
 * sequential order, and the state machine can be tested by executing operations
 * sequentially.
 *
 * Transitions:
 *  - `resume()`: PAUSED, PAUSING -> RUNNING
 *  - `pause_nb()`, `pause()`: RUNNING -> PAUSING
 *  - `stop()`: RUNNING, PAUSING, PAUSED -> STOPPING, FAILED -> STOPPED
 *  - `next_step()` (loop thread): PAUSING -> PAUSED, STOPPING -> STOPPED
 *  - `fail()` (loop thread): RUNNING, PAUSING -> FAILED, STOPPING -> STOPPED
 *
 * A failure is recorded in the same critical section as its transition, which leaves
 * the running states (RUNNING, PAUSING, STOPPING), and a failed loop never returns to
 * them.
 */
class LoopState {
  public:
    /// @brief The state of the loop.
    enum class State : std::uint8_t {
        Stopped,  ///< Stopped (terminal).
        Stopping,  ///< Stopping, the loop thread moves it to Stopped.
        Paused,  ///< Paused, can be resumed.
        Pausing,  ///< Pausing, the loop thread moves it to Paused.
        Running,  ///< Running the loop function.
        Failed,  ///< The loop function threw, can only be stopped.
    };

    /// @brief What the loop thread must do next, see `next_step()`.
    enum class Step : std::uint8_t {
        Run,  ///< Run the loop function.
        Exit,  ///< Exit the loop thread.
    };

    /**
     * @brief Get the current state.
     *
     * @return The state.
     */
    [[nodiscard]] State state() const noexcept;

    /**
     * @brief Check whether the loop is running.
     *
     * @return True if RUNNING, PAUSING or STOPPING.
     */
    [[nodiscard]] bool is_running() const noexcept;

    /**
     * @brief Resume the loop.
     *
     * @return True if the state changed to RUNNING.
     */
    bool resume() noexcept;

    /**
     * @brief Request the loop to pause, without waiting for it.
     *
     * @return True if the state changed to PAUSING.
     */
    bool pause_nb() noexcept;

    /**
     * @brief Request the loop to pause, and wait until it is no longer PAUSING.
     */
    void pause() noexcept;

    /**
     * @brief Stop the loop.
     *
     * @return True if this call initiated the stop, so the caller must join the loop
     * thread.
     */
    bool stop() noexcept;

    /**
     * @brief Record a failure of the loop function and leave the running states.
     *
     * Called on the loop thread only, at most once.
     *
     * @param error The exception thrown by the loop function.
     */
    void fail(std::exception_ptr error) noexcept;

    /**
     * @brief Get the failure of the loop function.
     *
     * @return The failure, or null if the loop function has not thrown.
     */
    [[nodiscard]] std::exception_ptr error() const noexcept;

    /**
     * @brief Wait until the loop thread must run the loop function or exit.
     *
     * Called on the loop thread only. Acknowledges pausing (PAUSING -> PAUSED) and
     * stopping (STOPPING -> STOPPED), and blocks while PAUSED or FAILED.
     *
     * @return The next step of the loop thread.
     */
    Step next_step() noexcept;

  private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;  ///< Notified on every state change.
    State state_{State::Paused};
    std::exception_ptr error_{};
};

}  // namespace rapidsmpf::detail
