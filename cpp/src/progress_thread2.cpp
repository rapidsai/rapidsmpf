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

#include <rapidsmpf/progress_thread2.hpp>

namespace rapidsmpf {

ProgressThread2::ProgressThread2(
    Communicator::Logger& logger, std::shared_ptr<Statistics> statistics
)
    : running_(false),
      next_task_id_(0),
      logger_(logger),
      statistics_(std::move(statistics)) {
    start();
}

ProgressThread2::~ProgressThread2() {
    stop();
}

void ProgressThread2::start() {
    if (running_)
        return;

    running_ = true;
    worker_thread_ = std::thread(&ProgressThread2::run, this);
}

void ProgressThread2::stop() {
    if (!running_) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        running_ = false;
    }
    cv_.notify_one();  // Wake up the worker thread to check if it should stop

    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
}

ProgressThread2::FunctionID ProgressThread2::add_function(Function task) {
    FunctionID id;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        id = next_task_id_++;
        staged_tasks_.emplace_back(id, std::move(task), ProgressState::InProgress);
    }
    cv_.notify_one();
    return id;
}

void ProgressThread2::remove_function(FunctionID id) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        staged_removals_.emplace_back(id);
    }
    cv_.notify_one();
}

void ProgressThread2::wait_for_function(FunctionID id) {
    std::unique_lock<std::mutex> lock(mutex_);

    // Wait until task is done or removed
    task_done_cv_.wait(lock, [this, id]() {
        return completed_tasks_.find(id) != completed_tasks_.end();
    });

    completed_tasks_.erase(id);
}

void ProgressThread2::clear_functions() {
    std::lock_guard<std::mutex> lock(mutex_);
    tasks_.clear();
    staged_tasks_.clear();
    staged_removals_.clear();
    completed_tasks_.clear();
}

void ProgressThread2::run() {
    while (true) {
        auto const t0_event_loop = Clock::now();
        // Merge staged tasks with main task list and handle removals
        {
            // lock the mutex to do the housekeeping work
            std::unique_lock lock(mutex_);
            // Wait until we should stop, or there are tasks to execute, or there are
            // staged changes to process
            cv_.wait(lock, [this] {
                return !running_ || !tasks_.empty() || !staged_tasks_.empty()
                       || !staged_removals_.empty();
            });

            if (!running_)
                break;

            // Handle staged removals
            for (FunctionID id : staged_removals_) {
                tasks_.erase(id);
                completed_tasks_.insert(id);
            }
            staged_removals_.clear();

            // Move staged tasks to main task list
            for (auto& task_info : staged_tasks_) {
                tasks_.emplace(task_info.id, std::move(task_info));
            }
            staged_tasks_.clear();
        }
        task_done_cv_.notify_all();  // wake up any waiting threads

        // Execute tasks and update their states
        for (auto it = tasks_.begin(); it != tasks_.end();) {
            ProgressState new_state = it->second.task();
            if (new_state == ProgressState::Done) {
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    completed_tasks_.insert(it->first);
                    it = tasks_.erase(it);
                }
                task_done_cv_.notify_all();  // wake up any waiting threads for completion
            } else {
                ++it;
            }
        }

        statistics_->add_duration_stat("event-loop-total", Clock::now() - t0_event_loop);
    }
}

}  // namespace rapidsmpf