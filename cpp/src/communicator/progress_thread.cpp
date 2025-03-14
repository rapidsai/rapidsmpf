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

#include <chrono>

#include <rapidsmp/communicator/progress_thread.hpp>
#include <rapidsmp/error.hpp>
#include <rapidsmp/utils.hpp>

#include "rapidsmp/pausable_thread_loop.hpp"

namespace rapidsmp {

ProgressThread::ProgressThread()
    : detail::PausableThreadLoop([this]() { return event_loop(this); }) {}

ProgressThread::~ProgressThread() {
    if (active_) {
        shutdown();
    }
}

void ProgressThread::shutdown() {
    RAPIDSMP_EXPECTS(active_, "ProgressThread is inactive");
    // auto& log = comm_->logger();
    // log.debug("Shuffler.shutdown() - initiate");
    event_loop_thread_run_.store(false);
    stop();
    // event_loop_thread_.join();
    // log.debug("Shuffler.shutdown() - done");
    active_ = false;
}

FunctionID ProgressThread::add_function(std::function<ProgressState()> function) {
    std::lock_guard const lock(mutex_);
    auto id = next_function_id_++;
    functions_.emplace_back(function, id);
    return id;
}

void ProgressThread::remove_function(FunctionID function_id) {
    auto state = std::find(functions_.begin(), functions_.end(), function_id);
    RAPIDSMP_EXPECTS(
        state != functions_.end(), "Iterable not registered or already removed"
    );
    {
        std::unique_lock state_lock(state->mutex);
        state->condition_variable.wait(state_lock, [state]() {
            return state->latest_state;
        });

        std::lock_guard const lock(mutex_);
        functions_.erase(state);
    }
}

void ProgressThread::event_loop(ProgressThread* self) {
    // auto& log = self->comm_->logger();

    // log.debug("event loop - start: ", *self);

    // This thread needs to have a cuda context associated with it.
    // For now, do so by calling cudaFree to initialise the driver.
    RAPIDSMP_CUDA_TRY(cudaFree(nullptr));
    // Continue the loop until both the "run" flag is false and all
    // ongoing communication is done.
    // auto const t0_event_loop = Clock::now();
    if (self->event_loop_thread_run_ || !self->functions_.empty()) {
        {
            std::lock_guard const lock(self->mutex_);
            for (auto& function : self->functions_) {
                function();
            }
        }
        // std::this_thread::yield();
    }
    // self->statistics_->add_duration_stat(
    //     "event-loop-total", Clock::now() - t0_event_loop
    // );
    // log.debug("event loop - shutdown: ", self->str());
}

}  // namespace rapidsmp
