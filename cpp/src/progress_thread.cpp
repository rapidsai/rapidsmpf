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

#include <utility>

#include <rapidsmp/error.hpp>
#include <rapidsmp/progress_thread.hpp>
#include <rapidsmp/utils.hpp>

#include "rapidsmp/communicator/communicator.hpp"

namespace rapidsmp {

ProgressThread::FunctionState::FunctionState(Function&& function)
    : function(std::move(function)) {}

void ProgressThread::FunctionState::operator()() {
    // Only call `function()` if it isn't done yet.
    if (!is_done.load() && function() == ProgressState::Done)
        is_done.store(true);
}

void ProgressThread::FunctionState::wait_for_completion(
    std::mutex& mutex, std::condition_variable& cv
) {
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [this]() { return is_done.load(); });
}

ProgressThread::ProgressThread(
    Communicator::Logger& logger, std::shared_ptr<Statistics> statistics
)
    : thread_([this]() {
          if (!is_thread_initialized_) {
              // This thread needs to have a cuda context associated with it.
              // For now, do so by calling cudaFree to initialise the driver.
              RAPIDSMP_CUDA_TRY(cudaFree(nullptr));
              is_thread_initialized_ = true;
          }
          return event_loop();
      }),
      logger_(logger),
      statistics_(std::move(statistics)) {}

ProgressThread::~ProgressThread() {
    if (active_.load()) {
        shutdown();
    }
}

void ProgressThread::shutdown() {
    RAPIDSMP_EXPECTS(active_.load(), "ProgressThread is inactive");
    logger_.debug("ProgressThread.shutdown() - initiate");
    event_loop_thread_run_.store(false);
    thread_.stop();
    logger_.debug("ProgressThread.shutdown() - done");
    active_.store(false);
}

ProgressThread::FunctionID ProgressThread::add_function(Function&& function) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto id =
        FunctionID(reinterpret_cast<ProgressThreadAddress>(this), next_function_id_++);
    functions_.emplace(id.function_index, std::move(function));
    thread_.resume();
    return id;
}

void ProgressThread::remove_function(FunctionID function_id) {
    RAPIDSMP_EXPECTS(
        function_id.thread_address == reinterpret_cast<ProgressThreadAddress>(this),
        "Function was not registered with this ProgressThread"
    );

    FunctionState* state = nullptr;
    {
        std::lock_guard const lock(mutex_);
        auto it = functions_.find(function_id.function_index);
        RAPIDSMP_EXPECTS(
            it != functions_.end(), "Function not registered or already removed"
        );
        state = &it->second;
    }

    // Wait for the function to complete
    state->wait_for_completion(state_mutex_, state_cv_);

    std::lock_guard const lock(mutex_);
    functions_.erase(function_id.function_index);

    if (functions_.empty()) {
        thread_.pause();
    }
}

void ProgressThread::event_loop() {
    auto const t0_event_loop = Clock::now();
    if (!event_loop_thread_run_.load()) {
        return;
    }

    {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        for (auto& [id, function] : functions_) {
            function();
        }
    }
    // Notify all waiting functions that we've completed an iteration
    {
        std::lock_guard const lock(state_mutex_);
        state_cv_.notify_all();
    }

    statistics_->add_duration_stat("event-loop-total", Clock::now() - t0_event_loop);
}

}  // namespace rapidsmp
