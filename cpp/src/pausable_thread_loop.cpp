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
            state_.wait(State::Paused);

            State expected = State::Pausing;
            if (state_.compare_exchange_strong(expected, State::Paused)) {
                state_.notify_all();
                continue;
            } else if (expected == State::Stopping) {
                state_.store(State::Stopped);
                state_.notify_all();
                return;
            } else if (expected == State::Paused) {
                continue;
            }  // else its Running

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
    return state_ != State::Paused;
}

void PausableThreadLoop::pause_nb() {
    State expected = State::Running;
    if (state_.compare_exchange_strong(expected, State::Pausing)) {
        state_.notify_all();
    }
}

void PausableThreadLoop::pause() {
    pause_nb();
    // And wait for the loop to flip to paused state. Behaviour is
    // undefined if someone else called resume/stop while we're in
    // this function.
    // std::unique_lock lock(mutex_);
    // cv_.wait(lock, [this]() { return state_ == State::Paused; });
    state_.wait(State::Pausing);
}

void PausableThreadLoop::resume() {
    while (true) {
        State expected = State::Paused;
        if (state_.compare_exchange_strong(expected, State::Running)) {
            state_.notify_all();
            return;
        } else if (expected == State::Running) {  // if its already running, we're done
            return;
        } else if (expected == State::Pausing) {
            // if its pausing, we need to wait for it to finish
            continue;
        } else {
            RAPIDSMPF_FAIL(
                "Unable to resume Pausable thread, because it is Stopped/Stopping."
            );
        }
    }
}

void PausableThreadLoop::stop() {
    // {
    //     std::lock_guard<std::mutex> lock(mutex_);
    //     state_ = State::Stopping;
    // }
    // cv_.notify_one();  // Wake up thread to exit

    if (state_ != State::Stopped) {
        state_.store(State::Stopping);
        state_.notify_all();

        state_.wait(State::Stopping);

        if (thread_.joinable()) {
            thread_.join();
        }
    }
}

}  // namespace rapidsmpf::detail
