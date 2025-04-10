/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmp/pausable_thread_loop.hpp>

namespace rapidsmp::detail {

PausableThreadLoop::PausableThreadLoop(std::function<void()> func, Duration sleep) {
    thread_ = std::thread([this, f = std::move(func), sleep]() {
        while (true) {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this]() { return !paused_ || !active_; });
                if (!active_) {
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
    return !paused_;
}

void PausableThreadLoop::pause() {
    std::lock_guard<std::mutex> lock(mutex_);
    paused_ = true;
}

void PausableThreadLoop::resume() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        paused_ = false;
    }
    cv_.notify_one();
}

void PausableThreadLoop::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        active_ = false;
        paused_ = false;  // Ensure it's not stuck in pause
    }
    cv_.notify_one();  // Wake up thread to exit
    if (thread_.joinable()) {
        thread_.join();
    }
}

}  // namespace rapidsmp::detail
