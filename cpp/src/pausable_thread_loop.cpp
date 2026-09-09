/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime_api.h>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/pausable_thread_loop.hpp>

namespace rapidsmpf::detail {

PausableThreadLoop::PausableThreadLoop(std::function<void()> func, Duration sleep) {
    thread_ = std::thread([this, f = std::move(func), sleep]() {
        // Establish a CUDA context on this thread before any user code runs. A
        // freshly spawned std::thread does not inherit the constructing thread's
        // CUDA context, and low-level driver-API calls (unlike the CUDA Runtime
        // API) do not lazily establish one on first use.
        RAPIDSMPF_CUDA_TRY(cudaFree(nullptr));

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

PausableThreadLoop::~PausableThreadLoop() noexcept {
    stop();
}

bool PausableThreadLoop::is_running() const noexcept {
    auto state = state_.load(std::memory_order_acquire);
    return state != State::Paused && state != State::Stopped;
}

void PausableThreadLoop::pause_nb() noexcept {
    State expected = State::Running;
    if (state_.compare_exchange_strong(
            expected, State::Pausing, std::memory_order_acq_rel, std::memory_order_relaxed
        ))
    {
        state_.notify_all();
    }
}

void PausableThreadLoop::pause() noexcept {
    pause_nb();
    state_.wait(State::Pausing, std::memory_order_acquire);
}

bool PausableThreadLoop::resume() noexcept {
    State curr = state_.load(std::memory_order_relaxed);
    while (curr == State::Paused || curr == State::Pausing) {
        // if state_ is still the value we saw, toggle it to Running
        if (state_.compare_exchange_weak(
                curr, State::Running, std::memory_order_acq_rel, std::memory_order_relaxed
            ))
        // using CAS weak because we can tolerate a spurious failure on retry
        {
            state_.notify_all();
            return true;
        }
    }
    return false;
}

bool PausableThreadLoop::stop() noexcept {
    State curr = state_.load(std::memory_order_relaxed);
    while (curr == State::Running || curr == State::Pausing || curr == State::Paused) {
        // if state_ is still the value we saw, toggle it to Stopping
        if (state_.compare_exchange_weak(
                curr,
                State::Stopping,
                std::memory_order_acq_rel,
                std::memory_order_relaxed
            ))
        // using CAS weak because we can tolerate a spurious failure on retry
        {
            state_.notify_all();

            // wait for state_ Stopping -> Stopped by the event loop thread
            if (thread_.joinable()) {
                thread_.join();
            }
            return true;
        }
        // else (someone other thread has changed state_) curr has the new value. retry.
    }
    return false;
}

}  // namespace rapidsmpf::detail
