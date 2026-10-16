/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <tuple>
#include <utility>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/progress_thread.hpp>
#include <rapidsmpf/utils/misc.hpp>

namespace rapidsmpf {

ProgressThread::FunctionState::FunctionState(
    Function&& function, ErrorCallback&& on_error
)
    : function(std::move(function)), on_error(std::move(on_error)) {}

void ProgressThread::FunctionState::operator()() {
    // Only call `function()` if it isn't done yet.
    // Note: ProgressThread::mutex_ is currently locked by the progress thread. So, we can
    // safely update is_done bool.
    if (is_done) {
        return;
    }
    try {
        if (function() == ProgressState::Done) {
            is_done = true;
        }
    } catch (...) {
        fail(std::current_exception());
    }
}

void ProgressThread::FunctionState::fail(std::exception_ptr error) noexcept {
    is_done = true;
    if (!on_error) {
        RAPIDSMPF_FATAL(
            "unhandled exception in ProgressThread function: "
            + detail::exception_message(error)
        );
    }
    try {
        on_error(std::move(error));
    } catch (...) {
        RAPIDSMPF_FATAL("ProgressThread error callback threw an exception");
    }
}

ProgressThread::ProgressThread(std::shared_ptr<Statistics> statistics, Duration sleep)
    : statistics_(std::move(statistics)),
      thread_(
          [this]() { return event_loop(); },
          sleep,
          [this](std::exception_ptr error) { handle_thread_error(std::move(error)); }
      ) {
    RAPIDSMPF_EXPECTS(statistics_ != nullptr, "the statistics pointer cannot be NULL");
}

ProgressThread::~ProgressThread() {
    stop();
}

void ProgressThread::stop() {
    thread_.stop();
}

ProgressThread::FunctionID ProgressThread::add_function(
    Function&& function, ErrorCallback&& on_error
) {
    std::lock_guard lock(mutex_);
    if (thread_error_) {
        std::rethrow_exception(thread_error_);
    }
    // We can use `this` as the thread address only because `ProgressThread` isn't
    // moveable or copyable.
    auto id =
        FunctionID(reinterpret_cast<ProgressThreadAddress>(this), next_function_id_++);
    functions_.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(id.function_index),
        std::forward_as_tuple(std::move(function), std::move(on_error))
    );
    thread_.resume();
    active_ = true;
    return id;
}

void ProgressThread::remove_function(FunctionID function_id) {
    RAPIDSMPF_EXPECTS(function_id.is_valid(), "FunctionID is not valid");
    RAPIDSMPF_EXPECTS(
        function_id.thread_address == reinterpret_cast<ProgressThreadAddress>(this),
        "Function was not registered with this ProgressThread"
    );

    std::unique_lock lock(mutex_);
    RAPIDSMPF_EXPECTS(
        functions_.contains(function_id.function_index),
        "Function not registered or already removed"
    );

    // Wait for the function to complete. If the thread is paused, then function will not
    // be able to complete, and therefore do not wait. Can't use map iterator because it
    // can get invalidated, if some other thread erases a function. So, query functions_
    // instead
    cv_.wait(lock, [&]() {
        return !active_ || functions_.at(function_id.function_index).is_done;
    });

    // Waiting done. Now, mutex_ is locked again
    functions_.erase(function_id.function_index);

    if (functions_.empty()) {
        thread_.pause_nb();
        active_ = false;
    }
    lock.unlock();
    cv_.notify_all();
}

void ProgressThread::pause() {
    thread_.pause();
    {
        std::lock_guard lock(mutex_);
        active_ = false;
    }
    cv_.notify_all();  // notify any thread waiting on active_
}

void ProgressThread::resume() {
    thread_.resume();
    {
        std::lock_guard lock(mutex_);
        // A failed thread loop cannot be resumed.
        active_ = thread_error_ == nullptr;
    }
    cv_.notify_all();
}

bool ProgressThread::is_running() const {
    std::lock_guard lock(mutex_);
    return active_;
}

std::shared_ptr<Statistics> ProgressThread::statistics() const noexcept {
    return statistics_;
}

void ProgressThread::event_loop() {
    auto const t0_event_loop = Clock::now();
    {
        std::lock_guard lock(mutex_);
        for (auto& [id, function] : functions_) {
            function();
        }
    }

    // Notify all waiting functions that we've completed an iteration
    cv_.notify_all();

    statistics_->add_duration_stat("event-loop-total", Clock::now() - t0_event_loop);
}

void ProgressThread::handle_thread_error(std::exception_ptr error) noexcept {
    {
        std::lock_guard lock(mutex_);
        thread_error_ = error;
        for (auto& [id, function] : functions_) {
            if (!function.is_done) {
                function.fail(error);
            }
        }
        // The underlying thread loop has failed and will not run again.
        active_ = false;
    }
    cv_.notify_all();
}

}  // namespace rapidsmpf
