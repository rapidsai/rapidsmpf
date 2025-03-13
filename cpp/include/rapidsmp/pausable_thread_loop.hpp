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

#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>

namespace rapidsmp::detail {
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
     * @note The loop starts paused.
     *
     * @param func The function to execute repeatedly in the thread.
     */
    PausableThreadLoop(std::function<void()> func) {
        thread_ = std::thread([this, func]() {
            while (true) {
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    cv_.wait(lock, [this]() { return !paused_ || !active_; });
                    if (!active_)
                        break;
                }
                func();
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });
    }

    /**
     * @brief Checks if the thread is currently running (not paused).
     * @return True if the thread is running, false if paused.
     */
    [[nodiscard]] bool is_running() const noexcept {
        return !paused_;
    }

    /**
     * @brief Pauses the execution of the thread.
     *
     * The thread will stop executing the func until `resume()` is called.
     */
    void pause() {
        std::lock_guard<std::mutex> lock(mutex_);
        paused_ = true;
    }

    /**
     * @brief Resumes execution of the thread after being paused.
     *
     * Calling resume on an already running loop is a no-op and is allowed.
     */
    void resume() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            paused_ = false;
        }
        cv_.notify_one();
    }

    /**
     * @brief Stops the execution of the thread and joins it.
     *
     * Once stopped, the thread cannot be resumed.
     */
    void stop() {
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

    /**
     * @brief Destructor that ensures the thread is stopped before destruction.
     */
    ~PausableThreadLoop() {
        stop();
    }

  private:
    std::thread thread_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool active_{true};
    bool paused_{true};
};

}  // namespace rapidsmp::detail
