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

ProgressThread::FunctionState::FunctionState(
    Function function,
    FunctionID function_id,
    std::mutex& mutex,
    std::condition_variable& cv
)
    : function(std::move(function)),
      function_id{std::move(function_id)},
      mutex_(mutex),
      cv_(cv) {}

ProgressThread::ProgressState ProgressThread::FunctionState::operator()() {
    ProgressState state = function();
    if (state == ProgressState::Done) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            is_done_ = true;
        }
        cv_.notify_all();
    }
    return state;
}

constexpr bool operator==(
    const ProgressThread::FunctionState& lhs, const ProgressThread::FunctionState& rhs
) {
    return lhs.function_id == rhs.function_id;
}

constexpr bool operator==(
    const ProgressThread::FunctionState& lhs,
    const ProgressThread::FunctionID& function_id
) {
    return lhs.function_id == function_id;
}

ProgressThread::ProgressThread(
    Communicator::Logger& logger, std::shared_ptr<Statistics> statistics
)
    : thread_([this]() {
          // This thread needs to have a cuda context associated with it.
          // For now, do so by calling cudaFree to initialise the driver.
          RAPIDSMP_CUDA_TRY(cudaFree(nullptr));
          return event_loop(this);
      }),
      logger_(logger),
      statistics_(std::move(statistics)) {}

ProgressThread::~ProgressThread() {
    if (active_) {
        shutdown();
    }
}

void ProgressThread::shutdown() {
    RAPIDSMP_EXPECTS(active_, "ProgressThread is inactive");
    logger_.debug("ProgressThread.shutdown() - initiate");
    event_loop_thread_run_.store(false);
    thread_.stop();
    logger_.debug("ProgressThread.shutdown() - done");
    active_ = false;
}

ProgressThread::FunctionID ProgressThread::add_function(
    std::function<ProgressState()> function
) {
    std::lock_guard const lock(mutex_);
    auto id = std::make_pair<std::uintptr_t, std::uint64_t>(
        reinterpret_cast<std::uintptr_t>(this), next_function_id_++
    );
    functions_.emplace_back(function, id, state_mutex_, state_cv_);
    thread_.resume();
    return id;
}

void ProgressThread::remove_function(FunctionID function_id) {
    decltype(functions_)::iterator state;

    RAPIDSMP_EXPECTS(
        function_id.first == reinterpret_cast<std::uintptr_t>(this),
        "Function was not registered with this ProgressThread"
    );

    {
        std::lock_guard const lock(mutex_);
        state = std::find(functions_.begin(), functions_.end(), function_id);
        RAPIDSMP_EXPECTS(
            state != functions_.end(), "Iterable not registered or already removed"
        );
    }

    // Wait for the function to complete
    state->wait_for_completion();

    {
        std::lock_guard const lock(mutex_);
        functions_.erase(state);
    }

    if (functions_.empty())
        thread_.pause();
}

void ProgressThread::event_loop(ProgressThread* self) {
    // Continue the loop until both the "run" flag is false and all
    // ongoing communication is done.
    auto const t0_event_loop = Clock::now();
    if (self->event_loop_thread_run_ || !self->functions_.empty()) {
        {
            std::lock_guard const lock(self->mutex_);
            for (auto& function : self->functions_) {
                function();
            }
        }
        // Notify all waiting functions that we've completed an iteration
        self->state_cv_.notify_all();
    }
    self->statistics_->add_duration_stat(
        "event-loop-total", Clock::now() - t0_event_loop
    );
}

}  // namespace rapidsmp
