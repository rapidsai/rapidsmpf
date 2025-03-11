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
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace rapidsmp {

/**
 * @brief Base for a class that can be iterated on by `ProgressThread`.
 *
 * Adds a single abstract method `progress()` that is called as part of the
 * `ProgressThread` to ensure the object is progressed in each iteration of
 * event loop.
 */
class ProgressThreadIterable {
  public:
    ProgressThreadIterable() = default;
    virtual ~ProgressThreadIterable() noexcept = default;

    /**
     * @brief Progress the object in the progress thread.
     *
     * @return Whether the object has completed progressing and can be destroyed.
     */
    virtual bool progress() = 0;
};

/**
 * @brief Store state of a `ProgressThreadIterable`.
 */
class ProgressThreadIterableState {
  public:
    ProgressThreadIterableState() = delete;

    /**
     * @brief Construct state of an iterable.
     *
     * @param iterable The raw pointer to the iterable instance.
     */
    ProgressThreadIterableState(rapidsmp::ProgressThreadIterable* iterable)
        : iterable{iterable} {}

    rapidsmp::ProgressThreadIterable* iterable;  ///< Pointer to the iterable instance.
    std::mutex mutex;  ///< Mutex to control access to state.
    std::condition_variable
        condition_variable;  ///< Condition variable to prevent early removal.
    bool latest_state{false};  ///< Latest progress state of iterable.
};

/**
 * @brief A progress thread that can execute arbitrary iteratables.
 *
 * Execute arbitrary iterables in a separate thread calling the `progress()`
 * method of each of the registered iterables. No ordering should be assumed
 * when progressing iterables, thus the iterables should not enforce any
 * dependency on the progress of other iterables.
 */
class ProgressThread {
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
    void insert_iterable(ProgressThreadIterable* iterable);

    /**
     * @brief Remove a iterable object and stop processing it as part of the event loop.
     *
     * @param iterable The iterable instance.
     */
    void erase_iterable(ProgressThreadIterable* iterable);

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
    std::unordered_map<
        ProgressThreadIterable*,
        std::unique_ptr<ProgressThreadIterableState>>
        iterables_;
    std::thread event_loop_thread_;
    std::atomic<bool> event_loop_thread_run_{true};
    std::mutex mutex_;
};

}  // namespace rapidsmp
