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
#include <cstdint>
#include <list>
#include <mutex>
#include <thread>

#include <rapidsmp/communicator/communicator.hpp>
#include <rapidsmp/pausable_thread_loop.hpp>
#include <rapidsmp/statistics.hpp>

namespace rapidsmp {

/**
 * @brief The progress state of a function, can be either `InProgress` or `Done`.
 */
enum ProgressState : bool {
    InProgress,
    Done,
};

/**
 * @typedef FunctionID
 * @brief The unique ID of a function registered with `ProgressThread`.
 */
using FunctionID = std::pair<std::uintptr_t, std::uint64_t>;

/**
 * @typedef Function
 * @brief The function type supported by `ProgressThread`, returning the progress
 * state of thee function.
 */
using Function = std::function<ProgressState()>;

/**
 * @brief Store state of a function.
 */
class FunctionState {
  public:
    FunctionState() = delete;

    /**
     * @brief Construct state of a function.
     *
     * @param function The function to execute.
     * @param function_id The ID that was assigned to the function by `ProgressThread`.
     */
    FunctionState(Function function, FunctionID function_id);

    /**
     * @brief Execute the function.
     *
     * @return The progress state of the function, that can be either
     * `ProgressState::InProgress` or `ProgressState::Done`.
     */
    ProgressState operator()();

    Function function;  ///< The function to execute.
    FunctionID
        function_id;  ///< The unique identifier of the function this object refers to.
    ProgressState latest_state{InProgress};  ///< Latest progress state of iterable.
};

/**
 * @brief Equality operator for a `FunctionState`.
 *
 * @param lhs The left hand side of the operator.
 * @param rhs The right hand side of the operator.
 *
 * @return Whether the objects point to the same function.
 */
constexpr bool operator==(const FunctionState& lhs, const FunctionState& rhs);

/**
 * @brief Equality operator for a `FunctionState`.
 *
 * @param lhs The left hand side of the operator.
 * @param function_id The function ID to compare to.
 *
 * @return Whether the objects point to the same function.
 */
constexpr bool operator==(const FunctionState& lhs, const FunctionID& function_id);

/**
 * @brief A progress thread that can execute arbitrary functions.
 *
 * Execute each of the registered arbitrary functions in a separate thread. The
 * functions are executed in the order they were registered, and a newly registered
 * function will only execute for the first time in the next iteration of the
 * progress thread.
 */
class ProgressThread {
  public:
    /**
     * @brief Construct a new progress thread that can handle multiple functions.
     *
     * @param logger The logger instance to use.
     * @param statistics The statistics instance to use (disabled by default).
     */
    ProgressThread(
        Communicator::Logger& logger,
        std::shared_ptr<Statistics> statistics = std::make_shared<Statistics>(false)
    );

    ~ProgressThread();

    /**
     * @brief Shutdown the thread, blocking until all functions are done.
     *
     * @throw std::logic_error If the thread is already inactive.
     */
    void shutdown();

    /**
     * @brief Insert an iterable object to process as part of the event loop.
     *
     * @param function The function to register.
     *
     * @return The unique ID of the function that was registered.
     */
    FunctionID add_function(std::function<ProgressState()> function);

    /**
     * @brief Remove a function and stop processing it as part of the event loop.
     *
     * @param function_id The unique function ID returned by `add_function`.
     *
     * @throws std::logic_error if the function was not registered with this
     * `ProgressThread` or was already removed.
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

    detail::PausableThreadLoop thread_;
    Communicator::Logger& logger_;
    std::shared_ptr<Statistics> statistics_;
    bool active_{true};
    std::list<FunctionState> functions_;
    std::thread event_loop_thread_;
    std::atomic<bool> event_loop_thread_run_{true};
    std::mutex mutex_;
    std::uint64_t next_function_id_;
};

}  // namespace rapidsmp
