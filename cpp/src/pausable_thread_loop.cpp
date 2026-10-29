/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>
#include <mutex>

#include <cuda_runtime_api.h>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/pausable_thread_loop.hpp>

#include "pausable_thread_loop_internal.hpp"

namespace rapidsmpf::detail {

namespace {

using State = LoopState::State;

/// @brief The loop owning the calling thread, if it is the thread of a loop.
thread_local PausableThreadLoop const* current_loop = nullptr;

}  // namespace

State LoopState::state() const noexcept {
    std::lock_guard lock(mutex_);
    return state_;
}

bool LoopState::is_running() const noexcept {
    std::lock_guard lock(mutex_);
    return state_ == State::Running || state_ == State::Pausing
           || state_ == State::Stopping;
}

bool LoopState::resume() noexcept {
    std::lock_guard lock(mutex_);
    if (state_ != State::Paused && state_ != State::Pausing) {
        return false;
    }
    state_ = State::Running;
    cv_.notify_all();
    return true;
}

bool LoopState::pause_nb() noexcept {
    std::lock_guard lock(mutex_);
    if (state_ != State::Running) {
        return false;
    }
    state_ = State::Pausing;
    cv_.notify_all();
    return true;
}

void LoopState::pause() noexcept {
    std::unique_lock lock(mutex_);
    if (state_ == State::Running) {
        state_ = State::Pausing;
        cv_.notify_all();
    }
    cv_.wait(lock, [this] { return state_ != State::Pausing; });
}

bool LoopState::stop() noexcept {
    std::lock_guard lock(mutex_);
    switch (state_) {
    case State::Running:
    case State::Pausing:
    case State::Paused:
        state_ = State::Stopping;
        break;
    case State::Failed:
        // A failed loop thread is idle, so the stop completes right away.
        state_ = State::Stopped;
        break;
    default:
        return false;
    }
    cv_.notify_all();
    return true;
}

void LoopState::fail(std::exception_ptr error) noexcept {
    std::lock_guard lock(mutex_);
    error_ = std::move(error);
    // While the loop function runs, the state is RUNNING, PAUSING or STOPPING.
    if (state_ == State::Stopping) {
        // Complete the stop, the `stop()` that moved the state to STOPPING joins.
        state_ = State::Stopped;
    } else {
        state_ = State::Failed;
    }
    cv_.notify_all();
}

std::exception_ptr LoopState::error() const noexcept {
    std::lock_guard lock(mutex_);
    return error_;
}

LoopState::Step LoopState::next_step() noexcept {
    std::unique_lock lock(mutex_);
    while (true) {
        switch (state_) {
        case State::Running:
            return Step::Run;
        case State::Pausing:
            state_ = State::Paused;
            cv_.notify_all();
            break;
        case State::Stopping:
            state_ = State::Stopped;
            cv_.notify_all();
            return Step::Exit;
        case State::Stopped:
            return Step::Exit;
        case State::Paused:
        case State::Failed:
            cv_.wait(lock);
            break;
        }
    }
}

PausableThreadLoop::PausableThreadLoop(
    std::function<void()> func, Duration sleep, ErrorHandler on_error
)
    : on_error_{std::move(on_error)}, state_{std::make_unique<LoopState>()} {
    thread_ = std::thread([this, f = std::move(func), sleep]() {
        current_loop = this;
        bool is_thread_initialized = false;

        while (true) {
            if (state_->next_step() == LoopState::Step::Exit) {
                return;
            }

            try {
                if (!is_thread_initialized) {
                    // Establish a CUDA context on this thread before the first user
                    // callback. A freshly spawned std::thread does not inherit the
                    // constructing thread's CUDA context, and low-level driver-API calls
                    // (unlike the CUDA Runtime API) do not lazily establish one on first
                    // use. Defer initialization until the loop is resumed so construction
                    // does not contend with CUDA setup by the calling thread.
                    RAPIDSMPF_CUDA_TRY(cudaFree(nullptr));
                    is_thread_initialized = true;
                }

                f();
            } catch (...) {
                // Never let an exception escape the thread, it would call
                // std::terminate() without giving the owner a chance to react.
                fail(std::current_exception());
                continue;
            }

            // Skip the sleep if the loop was paused or stopped during the iteration, for
            // example by the loop function itself.
            if (state_->state() != LoopState::State::Running) {
                continue;
            }
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
    RAPIDSMPF_EXPECTS_FATAL(
        current_loop != this, "PausableThreadLoop destroyed from its own thread"
    );
    stop();
    // The thread isn't joined by `stop()` if the loop was stopped from its own thread.
    if (thread_.joinable()) {
        thread_.join();
    }
}

void PausableThreadLoop::fail(std::exception_ptr error) noexcept {
    if (!on_error_) {
        RAPIDSMPF_FATAL(
            "unhandled exception in PausableThreadLoop: "
            + detail::exception_message(error)
        );
    }
    state_->fail(error);
    try {
        on_error_(std::move(error));
    } catch (...) {
        RAPIDSMPF_FATAL("PausableThreadLoop error handler threw an exception");
    }
}

std::exception_ptr PausableThreadLoop::error() const noexcept {
    return state_->error();
}

bool PausableThreadLoop::is_running() const noexcept {
    return state_->is_running();
}

void PausableThreadLoop::pause_nb() noexcept {
    state_->pause_nb();
}

void PausableThreadLoop::pause() noexcept {
    state_->pause();
}

bool PausableThreadLoop::resume() noexcept {
    return state_->resume();
}

bool PausableThreadLoop::stop() noexcept {
    if (!state_->stop()) {
        return false;
    }
    // A thread cannot join itself, the destructor joins it instead.
    if (current_loop != this && thread_.joinable()) {
        thread_.join();
    }
    return true;
}

}  // namespace rapidsmpf::detail
