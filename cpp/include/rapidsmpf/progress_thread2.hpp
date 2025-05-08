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

#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/statistics.hpp>

namespace rapidsmpf {

/**
 * @brief A class that manages a background thread that executes tasks
 */
class ProgressThread2 {
  public:
    /// @brief The state of a task
    enum class ProgressState {
        InProgress,
        Done
    };

    /// @brief The type of a task
    using Function = std::function<ProgressState()>;

    /// @brief The type of a task ID
    using FunctionID = uint64_t;

    /// @brief Constructor
    /// @param logger The logger to use
    /// @param statistics The statistics to use
    ProgressThread2(
        Communicator::Logger& logger,
        std::shared_ptr<Statistics> statistics = std::make_shared<Statistics>(false)
    );
    ~ProgressThread2();

    // Delete copy and move operations to make the class non-copyable and non-movable
    ProgressThread2(const ProgressThread2&) = delete;
    ProgressThread2& operator=(const ProgressThread2&) = delete;
    ProgressThread2(ProgressThread2&&) = delete;
    ProgressThread2& operator=(ProgressThread2&&) = delete;

    /// @brief Start the background thread that executes tasks
    void start();

    /// @brief Stop the background thread and wait for it to finish
    void stop();

    /// @brief Add a new task to the queue
    /// @param task The task to add
    /// @return A unique ID that can be used to wait for the task
    FunctionID add_function(Function task);

    /// @brief Remove a task by its ID
    /// @param id The ID of the task to remove
    /// @note This function returns immediately, the task will be removed by the worker
    /// thread.
    void remove_function(FunctionID id);

    /// @brief Wait for a task to complete
    /// @param id The ID of the task to wait for
    /// @note This method will block until the task is completed and removed by the
    /// background thread If the task is not found, returns immediately
    void wait_for_function(FunctionID id);

    /// @brief Remove all tasks immediately
    /// This will not wait for tasks to complete
    void clear_functions();

  private:
    /// @brief The information about a task
    struct FunctionInfo {
        FunctionID id;  ///< The ID of the task
        Function task;  ///< The task to execute
        ProgressState state;  ///< The state of the task

        FunctionInfo(FunctionID id, Function task, ProgressState state)
            : id(id), task(std::move(task)), state(state) {}
    };

    /// @brief The main loop of the background thread
    void run();

    std::unordered_map<FunctionID, FunctionInfo> tasks_;  ///< Main task list
    std::vector<FunctionInfo> staged_tasks_;  ///< Staging area for new tasks
    std::vector<FunctionID>
        staged_removals_;  ///< Staging area for task IDs to be removed
    std::unordered_set<FunctionID>
        completed_tasks_;  ///< Set of completed tasks (done/ removed)
    std::mutex mutex_;
    std::condition_variable cv_;  ///< For waiting on new tasks
    std::condition_variable task_done_cv_;  ///< For waiting on task completion
    std::atomic<bool> running_;
    std::atomic<FunctionID> next_task_id_;
    std::thread worker_thread_;
    Communicator::Logger& logger_;
    std::shared_ptr<Statistics> statistics_;
};

}  // namespace rapidsmpf
