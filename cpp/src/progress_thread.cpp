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

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/progress_thread.hpp>
#include <rapidsmpf/utils.hpp>

#include "rapidsmpf/communicator/communicator.hpp"

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

ProgressThread::ProgressThread(
    Communicator::Logger& logger, std::shared_ptr<Statistics> statistics, Duration sleep
)
    : thread_(
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
    ),
      logger_(logger),
      statistics_(std::move(statistics)) {
    RAPIDSMPF_EXPECTS(statistics_ != nullptr, "the statistics pointer cannot be NULL");
}

ProgressThread::~ProgressThread() {
    stop();
}

void ProgressThread::stop() {
    logger_.debug("ProgressThread.stop() - initiate");
    thread_.stop();
    logger_.debug("ProgressThread.stop() - done");
}

ProgressThread::FunctionID ProgressThread::add_function(Function&& function) {
    std::lock_guard lock(staging_mutex_);
    auto id =
        FunctionID(reinterpret_cast<ProgressThreadAddress>(this), next_function_id_++);
    staged_functions_.emplace(id.function_index, std::move(function));
    thread_.resume();
    return id;
}

void ProgressThread::remove_function(FunctionID function_id) {
    RAPIDSMPF_EXPECTS(function_id.is_valid(), "FunctionID is not valid");
    RAPIDSMPF_EXPECTS(
        function_id.thread_address == reinterpret_cast<ProgressThreadAddress>(this),
        "Function was not registered with this ProgressThread"
    );

    auto function_idx = function_id.function_index;

    // First check if it's already completed
    if (completion_tracker_.is_completed(function_idx)) {
        completion_tracker_.remove(function_idx);
        return;
    }

    // Wait for completion
    completion_tracker_.wait_for_completion(function_idx);
    completion_tracker_.remove(function_idx);
}

void ProgressThread::event_loop() {
    auto const t0_event_loop = Clock::now();

    // Move any staged functions to active functions
    {
        std::lock_guard<std::mutex> lock(staging_mutex_);
        if (!staged_functions_.empty()) {
            functions_.insert(staged_functions_.begin(), staged_functions_.end());
            staged_functions_.clear();
        }
    }

    // Process all active functions without holding any locks
    std::vector<FunctionIndex> completed_functions;
    for (auto& [id, function] : functions_) {
        function();
        if (function.is_done) {
            completed_functions.push_back(id);
            completion_tracker_.mark_completed(id);
        }
    }

    // Remove completed functions and check if we should pause
    if (!completed_functions.empty()) {
        std::lock_guard<std::mutex> lock(staging_mutex_);
        for (auto id : completed_functions) {
            functions_.erase(id);
        }

        if (functions_.empty() && staged_functions_.empty()) {
            thread_.pause();
        }
    }

    statistics_->add_duration_stat("event-loop-total", Clock::now() - t0_event_loop);
}

}  // namespace rapidsmpf
