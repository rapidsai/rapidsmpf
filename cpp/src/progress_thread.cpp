/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <utility>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/progress_thread.hpp>
#include <rapidsmpf/utils/misc.hpp>

namespace rapidsmpf {

ProgressThread::FunctionState::FunctionState(Function&& function)
    : function(std::move(function)) {}

void ProgressThread::FunctionState::operator()() {
    // Only call `function()` if it isn't done yet.
    // Note: ProgressThread::mutex_ is currently locked by the progress thread. So, we can
    // safely update is_done bool.
    if (!is_done && function() == ProgressState::Done) {
        is_done = true;
    }
}

ProgressThread::ProgressThread(std::shared_ptr<Statistics> statistics, Duration sleep)
    : statistics_(std::move(statistics)),
      thread_(
          [this]() {
              if (!is_thread_initialized_) {
                  // This thread needs to have a cuda context associated with it.
                  // For now, do so by calling cudaFree to initialise the driver.
                  RAPIDSMPF_CUDA_TRY(cudaFree(nullptr));
                  is_thread_initialized_ = true;
              }
              return event_loop();
          },
          sleep
      ) {
    RAPIDSMPF_EXPECTS(statistics_ != nullptr, "the statistics pointer cannot be NULL");
}

ProgressThread::~ProgressThread() {
    stop();
}

void ProgressThread::stop() {
    thread_.stop();
}

ProgressThread::FunctionID ProgressThread::add_function(Function&& function) {
    std::lock_guard lock(mutex_);
    // We can use `this` as the thread address only because `ProgressThread` isn't
    // moveable or copyable.
    auto id =
        FunctionID(reinterpret_cast<ProgressThreadAddress>(this), next_function_id_++);
    functions_.emplace(id.function_index, std::move(function));
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
        active_ = true;
    }
    cv_.notify_all();
}

bool ProgressThread::is_running() const {
    std::lock_guard lock(mutex_);
    return active_;
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

}  // namespace rapidsmpf
