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
        bool is_thread_initialized = false;

        while (true) {
            // wait until the thread is not paused
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [&] { return state_ != State::Paused; });

            if (state_ == State::Pausing) {
                state_ = State::Paused;
                cv_.notify_all();
                continue;
            }
            if (state_ == State::Stopping) {
                state_ = State::Stopped;
                cv_.notify_all();
                return;
            }

            // This iteration serves any pending resume(). A resume() during the
            // iteration or the sleep sets SkipSleep again, which skips or ends the sleep.
            state_ = State::Running;
            lock.unlock();  // Never run the loop function while holding the lock.

            if (!is_thread_initialized) {
                // Establish a CUDA context on this thread before the first user callback.
                // A freshly spawned std::thread does not inherit the constructing
                // thread's CUDA context, and low-level driver-API calls (unlike the CUDA
                // Runtime API) do not lazily establish one on first use. Defer
                // initialization until the loop is resumed so construction does not
                // contend with CUDA setup by the calling thread.
                RAPIDSMPF_CUDA_TRY(cudaFree(nullptr));
                is_thread_initialized = true;
            }

            f();

            if (sleep > std::chrono::seconds{0}) {
                auto const deadline = std::chrono::steady_clock::now() + sleep;
                lock.lock();
                cv_.wait_until(lock, deadline, [&] { return state_ != State::Running; });
                lock.unlock();
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
    std::lock_guard<std::mutex> lock(mutex_);
    return state_ != State::Paused && state_ != State::Stopped;
}

void PausableThreadLoop::pause_nb() noexcept {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ != State::Running && state_ != State::SkipSleep) {
            return;
        }
        state_ = State::Pausing;
    }
    cv_.notify_all();
}

void PausableThreadLoop::pause() noexcept {
    pause_nb();
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&] { return state_ != State::Pausing; });
}

bool PausableThreadLoop::resume() noexcept {
    bool was_paused;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ == State::Stopping || state_ == State::Stopped
            || state_ == State::SkipSleep)
        {
            return false;
        }
        was_paused = state_ == State::Paused || state_ == State::Pausing;
        state_ = State::SkipSleep;  // event loop will switch to Running
    }
    cv_.notify_all();
    return was_paused;
}

bool PausableThreadLoop::stop() noexcept {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ == State::Stopping || state_ == State::Stopped) {
            return false;
        }
        state_ = State::Stopping;
    }
    cv_.notify_all();

    // wait for state_ Stopping -> Stopped by the event loop thread
    if (thread_.joinable()) {
        thread_.join();
    }
    return true;
}

}  // namespace rapidsmpf::detail
