/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/pausable_thread_loop.hpp>

namespace rapidsmpf::detail {

PausableThreadLoop::PausableThreadLoop(std::function<void()> func, Duration sleep) {
    thread_ = std::thread([this, f = std::move(func), sleep]() {
        while (true) {
            // wait until the thread is not paused
            state_.wait(State::Paused, std::memory_order_acquire);

            // if the thread is pausing, set it to paused, and loop again
            State expected = State::Pausing;
            if (state_.compare_exchange_strong(
                    expected,
                    State::Paused,
                    std::memory_order_acq_rel,
                    std::memory_order_relaxed
                ))
            {
                state_.notify_all();
                continue;
            }  // else - We don't have to worry about Stopped state, because Stopping ->
               // Stopped state change only performed by this thread.

            expected = State::Stopping;
            if (state_.compare_exchange_strong(
                    expected,
                    State::Stopped,
                    std::memory_order_acq_rel,
                    std::memory_order_relaxed
                ))
            {
                state_.notify_all();
                return;
            }

            f();

            if (sleep > std::chrono::seconds{0}) {
                std::this_thread::sleep_for(sleep);
            } else {
                std::this_thread::yield();
            }
            // Add a short sleep to avoid other threads starving under Valgrind.
            if (is_running_under_valgrind()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
    });
}

PausableThreadLoop::~PausableThreadLoop() {
    stop();
}

bool PausableThreadLoop::is_running() const noexcept {
    auto state = state_.load(std::memory_order_acquire);
    return state != State::Paused && state != State::Stopped;
}

void PausableThreadLoop::pause_nb() {
    State expected = State::Running;
    if (state_.compare_exchange_strong(
            expected, State::Pausing, std::memory_order_acq_rel, std::memory_order_relaxed
        ))
    {
        state_.notify_all();
    }
}

void PausableThreadLoop::pause() {
    pause_nb();
    state_.wait(State::Pausing, std::memory_order_acquire);
}

void PausableThreadLoop::resume() {
    State curr = state_.load(std::memory_order_relaxed);
    while (curr != State::Stopping && curr != State::Stopped && curr != State::Running) {
        // if state_ is still the value we saw, toggle it to Running
        // (possible values for curr: Paused, Pausing)
        if (state_.compare_exchange_weak(
                curr, State::Running, std::memory_order_acq_rel, std::memory_order_relaxed
            ))
        // using CAS weak because we can tolerate a spurious failure on retry
        {
            state_.notify_all();
            return;
        }
    }
}

void PausableThreadLoop::stop() {
    State curr = state_.load(std::memory_order_relaxed);
    while (curr != State::Stopping && curr != State::Stopped) {
        // if state_ is still the value we saw, toggle it to Stopping
        // (possible values for curr: Paused, Pausing, Running)
        if (state_.compare_exchange_weak(
                curr,
                State::Stopping,
                std::memory_order_acq_rel,
                std::memory_order_relaxed
            ))
        // using CAS weak because we can tolerate a spurious failure on retry
        {
            state_.notify_all();

            // wait for state_ Stopping -> Stopped
            state_.wait(State::Stopping, std::memory_order_acquire);

            if (thread_.joinable()) {
                thread_.join();
            }
            return;
        }
        // else (someone other thread has changed state_) curr has the new value. retry.
    }
}

}  // namespace rapidsmpf::detail
