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

namespace rapidsmp {

ProgressThread::ProgressThread() {
    event_loop_thread_ = std::thread(ProgressThread::event_loop, this);
}

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
    event_loop_thread_.join();
    // log.debug("Shuffler.shutdown() - done");
    active_ = false;
}

void ProgressThread::insert_iterable(ProgressThreadIterable* iterable) {
    auto it = iterables_.find(iterable);
    RAPIDSMP_EXPECTS(it == iterables_.end(), "Iterable already registered");
    std::lock_guard const lock(mutex_);
    iterables_.insert({iterable, std::make_unique<ProgressThreadIterableState>(iterable)}
    );
}

void ProgressThread::erase_iterable(ProgressThreadIterable* iterable) {
    auto it = iterables_.find(iterable);
    RAPIDSMP_EXPECTS(
        it != iterables_.end(), "Iterable not registered or already removed"
    );
    auto iterable_state = it->second.get();
    {
        std::unique_lock iterable_lock(iterable_state->mutex);
        iterable_state->condition_variable.wait(iterable_lock, [iterable_state]() {
            return iterable_state->latest_state;
        });

        std::lock_guard const lock(mutex_);
        iterables_.erase(iterable);
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
    while (self->event_loop_thread_run_ || !self->iterables_.empty()) {
        {
            // if (!self->event_loop_thread_run_) continue;
            std::lock_guard const lock(self->mutex_);
            for (auto& iterable_pair : self->iterables_) {
                auto iterable = iterable_pair.first;
                auto iterable_state = iterable_pair.second.get();
                iterable_state->latest_state = iterable->progress();
                if (iterable_state->latest_state)
                    iterable_state->condition_variable.notify_all();
            }
        }
        std::this_thread::yield();

        // Let's add a short sleep to avoid other threads starving under Valgrind.
        if (is_running_under_valgrind()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    // self->statistics_->add_duration_stat(
    //     "event-loop-total", Clock::now() - t0_event_loop
    // );
    // log.debug("event loop - shutdown: ", self->str());
}

}  // namespace rapidsmp
