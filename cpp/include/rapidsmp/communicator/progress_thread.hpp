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
#include <list>
#include <mutex>
#include <thread>
#include <utility>

#include <rapidsmp/pausable_thread_loop.hpp>

namespace rapidsmp {

enum ProgressState : bool {
    InProgress,
    Done,
};

using FunctionID = std::uint64_t;
using Function = std::function<ProgressState()>;

/**
 * @brief Store state of a `ProgressThreadIterable`.
 */
class FunctionState {
  public:
    FunctionState() = delete;

    /**
     * @brief Construct state of an iterable.
     *
     * @param iterable The raw pointer to the iterable instance.
     */
    FunctionState(Function function, FunctionID function_id)
        : function(std::move(function)), function_id{function_id} {}

    ProgressState operator()() {
        std::lock_guard const lock(mutex);
        latest_state = function();
        if (latest_state == ProgressState::Done)
            condition_variable.notify_all();
        return latest_state;
    }

    Function function;
    FunctionID
        function_id;  ///< The unique identifier of the function this object refers to.
    std::mutex mutex;  ///< Mutex to control access to state.
    std::condition_variable
        condition_variable;  ///< Condition variable to prevent early removal.
    ProgressState latest_state{InProgress};  ///< Latest progress state of iterable.
};

constexpr bool operator==(const FunctionState& lhs, const FunctionState& rhs) {
    return lhs.function_id == rhs.function_id;
}

constexpr bool operator==(const FunctionState& lhs, const FunctionID& function_id) {
    return lhs.function_id == function_id;
}

/**
 * @brief A progress thread that can execute arbitrary iteratables.
 *
 * Execute arbitrary iterables in a separate thread calling the `progress()`
 * method of each of the registered iterables. No ordering should be assumed
 * when progressing iterables, thus the iterables should not enforce any
 * dependency on the progress of other iterables.
 */
class ProgressThread : public detail::PausableThreadLoop {
  public:
    /**
     * @brief Construct a new progress thread that can handle multiple iterables.
     */
    ProgressThread();

    ~ProgressThread();

    /**
     * @brief Shutdown the thread, blocking until all iterables are done.
     *
     * @throw std::logic_error If the thread is already inactive.
     */
    void shutdown();

    /**
     * @brief Insert an iterable object to process as part of the event loop.
     *
     * @param iterable The iterable instance.
     */
    FunctionID add_function(std::function<ProgressState()> function);

    /**
     * @brief Remove a iterable object and stop processing it as part of the event loop.
     *
     * @param iterable The iterable instance.
     */
    void remove_function(FunctionID function_id);

  private:
    /**
     * @brief The event loop progressing each of the iterables.
     *
     * The event loop continuously progresses registered iterables in no
     * specific order.
     *
     * @param self The `ProgressThread` instance.
     */
    static void event_loop(ProgressThread* self);

    bool active_{true};
    std::list<FunctionState> functions_;
    std::thread event_loop_thread_;
    std::atomic<bool> event_loop_thread_run_{true};
    std::mutex mutex_;
    FunctionID next_function_id_;
};

}  // namespace rapidsmp
