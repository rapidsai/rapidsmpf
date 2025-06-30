/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/pausable_thread_loop.hpp>

namespace rapidsmpf::detail {

PausableThreadLoop::PausableThreadLoop(std::function<void()> func, Duration sleep) {
    thread_ = std::thread([this, f = std::move(func), sleep]() {
        while (true) {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this]() { return state_ != State::Paused; });
                // Note: state changes here must notify any waiters.
                switch (state_) {
                case State::Running:
                    break;
                case State::Pausing:
                    state_ = State::Paused;
                    lock.unlock();
                    cv_.notify_one();
                case State::Paused:
                    continue;
                case State::Stopping:
                    state_ = State::Stopped;
                    lock.unlock();
                    cv_.notify_one();
                case State::Stopped:
                    return;
                }
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
    std::lock_guard<std::mutex> lock(mutex_);
    return state_ != State::Paused;
}

void PausableThreadLoop::pause_nb() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        state_ = State::Pausing;
    }
    cv_.notify_one();
}

void PausableThreadLoop::pause() {
    pause_nb();
    // And wait for the loop to flip to paused state. Behaviour is
    // undefined if someone else called resume/stop while we're in
    // this function.
    std::unique_lock lock(mutex_);
    cv_.wait(lock, [this]() { return state_ == State::Paused; });
}

void PausableThreadLoop::resume() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        state_ = State::Running;
    }
    cv_.notify_one();
}

void PausableThreadLoop::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        state_ = State::Stopping;
    }
    cv_.notify_one();  // Wake up thread to exit
    if (thread_.joinable()) {
        thread_.join();
    }
}

}  // namespace rapidsmpf::detail
