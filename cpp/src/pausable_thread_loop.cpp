/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <rapidsmp/pausable_thread_loop.hpp>

namespace rapidsmp::detail {

PausableThreadLoop::PausableThreadLoop(
    std::function<void()> func, std::chrono::microseconds sleep
) {
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
            if (sleep > std::chrono::microseconds(0)) {
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
