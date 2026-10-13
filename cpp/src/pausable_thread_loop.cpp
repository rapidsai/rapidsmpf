/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime_api.h>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/pausable_thread_loop.hpp>

namespace rapidsmpf::detail {

namespace {

using State = LoopState::State;

/// @brief The loop owning the calling thread, if it is the thread of a loop.
thread_local PausableThreadLoop const* current_loop = nullptr;

std::optional<State> on_resume(State s) noexcept {
    if (s == State::Paused || s == State::Pausing) {
        return State::Running;
    }
    return std::nullopt;
}

std::optional<State> on_pause(State s) noexcept {
    if (s == State::Running) {
        return State::Pausing;
    }
    return std::nullopt;
}

std::optional<State> on_stop(State s) noexcept {
    switch (s) {
    case State::Running:
    case State::Pausing:
    case State::Paused:
        return State::Stopping;
    case State::Failed:
        // A failed loop thread is idle, so the stop completes right away.
        return State::Stopped;
    default:
        return std::nullopt;
    }
}

std::optional<State> on_step(State s) noexcept {
    switch (s) {
    case State::Pausing:
        return State::Paused;
    case State::Stopping:
        return State::Stopped;
    default:
        return std::nullopt;
    }
}

std::optional<State> on_fail(State s) noexcept {
    switch (s) {
    case State::Running:
    case State::Pausing:
        return State::Failed;
    case State::Stopping:
        // Complete the stop, the `stop()` that moved the state to STOPPING joins.
        return State::Stopped;
    default:
        return std::nullopt;
    }
}

}  // namespace

std::optional<State> LoopState::apply(Transition transition) noexcept {
    State curr = state_.load(std::memory_order_acquire);
    while (auto next = transition(curr)) {
        // using CAS weak because we can tolerate a spurious failure on retry
        if (state_.compare_exchange_weak(
                curr, *next, std::memory_order_acq_rel, std::memory_order_acquire
            ))
        {
            state_.notify_all();
            return next;
        }
    }
    return std::nullopt;
}

State LoopState::state() const noexcept {
    return state_.load(std::memory_order_acquire);
}

bool LoopState::is_running() const noexcept {
    auto const s = state();
    return s == State::Running || s == State::Pausing || s == State::Stopping;
}

bool LoopState::resume() noexcept {
    return apply(on_resume).has_value();
}

bool LoopState::pause_nb() noexcept {
    return apply(on_pause).has_value();
}

void LoopState::wait_while_pausing() const noexcept {
    state_.wait(State::Pausing, std::memory_order_acquire);
}

bool LoopState::stop() noexcept {
    return apply(on_stop).has_value();
}

void LoopState::fail(std::exception_ptr error) noexcept {
    error_ = std::move(error);
    apply(on_fail);
    // Publish `error_` only after leaving the running states. A thread observing
    // `failed_` happens-after the transition, so it observes FAILED or STOPPED.
    failed_.store(true, std::memory_order_release);
}

std::exception_ptr LoopState::error() const noexcept {
    if (failed_.load(std::memory_order_acquire)) {
        return error_;
    }
    return {};
}

LoopState::Step LoopState::step() noexcept {
    if (auto next = apply(on_step)) {
        return *next == State::Stopped ? Step::Exit : Step::Wait;
    }
    switch (state()) {
    case State::Running:
        return Step::Run;
    case State::Stopped:
        return Step::Exit;
    default:
        return Step::Wait;
    }
}

void LoopState::wait_while_idle() const noexcept {
    for (auto s = state(); s == State::Paused || s == State::Failed; s = state()) {
        state_.wait(s, std::memory_order_acquire);
    }
}

PausableThreadLoop::PausableThreadLoop(
    std::function<void()> func, Duration sleep, ErrorHandler on_error
)
    : on_error_{std::move(on_error)} {
    thread_ = std::thread([this, f = std::move(func), sleep]() {
        current_loop = this;
        bool is_thread_initialized = false;

        while (true) {
            state_.wait_while_idle();
            switch (state_.step()) {
            case LoopState::Step::Wait:
                continue;
            case LoopState::Step::Exit:
                return;
            case LoopState::Step::Run:
                break;
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
            if (state_.state() != LoopState::State::Running) {
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
    state_.fail(error);
    try {
        on_error_(std::move(error));
    } catch (...) {
        RAPIDSMPF_FATAL("PausableThreadLoop error handler threw an exception");
    }
}

std::exception_ptr PausableThreadLoop::error() const noexcept {
    return state_.error();
}

bool PausableThreadLoop::is_running() const noexcept {
    return state_.is_running();
}

void PausableThreadLoop::pause_nb() noexcept {
    state_.pause_nb();
}

void PausableThreadLoop::pause() noexcept {
    state_.pause_nb();
    state_.wait_while_pausing();
}

bool PausableThreadLoop::resume() noexcept {
    return state_.resume();
}

bool PausableThreadLoop::stop() noexcept {
    if (!state_.stop()) {
        return false;
    }
    // A thread cannot join itself, the destructor joins it instead.
    if (current_loop != this && thread_.joinable()) {
        thread_.join();
    }
    return true;
}

}  // namespace rapidsmpf::detail
