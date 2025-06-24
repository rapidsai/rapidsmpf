/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <condition_variable>  // NOLINT(unused-includes)
#include <iostream>
#include <mutex>
#include <thread>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/utils.hpp>

namespace rapidsmpf {

#define RAPIDSMPF_DEBUG

/**
 * @def rapidsmpf_mutex_t
 * @brief Type alias for mutex used throughout RAPIDSMPF.
 *
 * In debug mode, this resolves to `std::timed_mutex` for deadlock detection.
 * In release mode, this resolves to a standard `std::mutex` for performance.
 */
#ifdef RAPIDSMPF_DEBUG
#define rapidsmpf_mutex_t std::timed_mutex
#else
#define rapidsmpf_mutex_t std::mutex
#endif

/**
 * @def rapidsmpf_condition_variable_t
 * @brief Type alias for condition variable used throughout RAPIDSMPF.
 *
 * In debug mode, this resolves to `std::condition_variable_any` for deadlock detection.
 * In release mode, this resolves to a standard `std::condition_variable` for performance.
 */
#ifdef RAPIDSMPF_DEBUG
#define rapidsmpf_condition_variable_t std::condition_variable_any
#else
#define rapidsmpf_condition_variable_t std::condition_variable
#endif

/**
 * @def RAPIDSMPF_LOCK_GUARD
 * @brief Lock guard macro that optionally enforces a timeout in debug builds.
 *
 * In debug mode, this uses a `timeout_lock_guard` to detect deadlocks.
 * In release mode, it uses a standard `std::lock_guard` for performance.
 *
 * @param mutex A reference to a `rapidsmpf_mutex_t` instance.
 */
#ifdef RAPIDSMPF_DEBUG
#define RAPIDSMPF_LOCK_GUARD(mutex)             \
    rapidsmpf::detail::timeout_lock_guard const \
        RAPIDSMPF_CONCAT(_rapidsmpf_lock_guard_, __LINE__)(mutex, __FILE__, __LINE__);
#else
#define RAPIDSMPF_LOCK_GUARD(mutex)          \
    std::lock_guard<rapidsmpf_mutex_t> const \
        RAPIDSMPF_CONCAT(_rapidsmpf_lock_guard_, __LINE__)(mutex);
#endif


namespace detail {

/**
 * @brief A lock guard that enforces a timeout when acquiring a lock.
 *
 * Used in debug builds to help detect potential deadlocks by failing fast
 * when a lock cannot be acquired within a fixed duration.
 *
 * @note Automatically unlocks the mutex when destroyed.
 */
class timeout_lock_guard {
  public:
    /**
     * @brief Constructs and attempts to acquire a lock with a timeout.
     *
     * If the lock cannot be acquired within the specified timeout, a
     * warning is written to stderr.
     *
     * @param mutex The `std::timed_mutex` to lock.
     * @param filename Source file name (used in error message).
     * @param line_number Line number (used in error message).
     * @param timeout Maximum duration to wait before timing out.
     *
     * @throws std::runtime_error If the lock could not be acquired in time.
     */
    timeout_lock_guard(
        std::timed_mutex& mutex,
        char const* filename,
        int line_number,
        Duration const& timeout = std::chrono::seconds{60}
    )
        : lock_(mutex, std::defer_lock) {
        while (!lock_.try_lock_for(timeout)) {
            std::stringstream ss;
            ss << "[DEADLOCK] timeout(" << timeout.count() << "s): " << filename << ":"
               << line_number;
            std::cerr << ss.str() << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(10));
        }
    }

    // No move or copy.
    timeout_lock_guard(const timeout_lock_guard&) = delete;
    timeout_lock_guard& operator=(const timeout_lock_guard&) = delete;
    timeout_lock_guard(timeout_lock_guard&&) = delete;
    timeout_lock_guard& operator=(timeout_lock_guard&&) = delete;

  private:
    std::unique_lock<std::timed_mutex> lock_;
};
}  // namespace detail

}  // namespace rapidsmpf
