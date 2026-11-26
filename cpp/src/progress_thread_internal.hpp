/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <string_view>
#include <utility>

#include <rapidsmpf/communicator/logger.hpp>
#include <rapidsmpf/error.hpp>

namespace rapidsmpf::detail {

/**
 * @brief Report the error that made the progress function of an operation fail.
 *
 * The `ProgressThread::ErrorCallback` of the operations driven by a `ProgressThread`
 * (shuffler, collectives). Records the first error in `error_slot`, wakes up the
 * threads waiting on `cv`, which then rethrow it, and invokes the finished callback so
 * that asynchronous waiters wake up too.
 *
 * An error raised after the operation completed, e.g., by a throwing finished callback,
 * is only logged: the result is complete, so it is not failed.
 *
 * @param error The exception that made the progress function fail.
 * @param mutex The mutex of the operation, which guards `completed` and `error_slot`.
 * @param cv The condition variable the threads waiting on the operation wait on.
 * @param completed Returns whether the operation completed, called with `mutex` held.
 * @param error_slot The error of the operation.
 * @param finished_callback The finished callback of the operation, consumed.
 * @param logger The logger of the operation.
 * @param operation The name of the operation, for the log.
 */
template <typename Completed>
void report_progress_error(
    std::exception_ptr error,
    std::mutex& mutex,
    std::condition_variable& cv,
    Completed const& completed,
    std::exception_ptr& error_slot,
    std::function<void()>& finished_callback,
    Logger& logger,
    std::string_view operation
) noexcept {
    // Never let a failure to log escape, which would terminate the process.
    auto const warn = [&](std::string_view what, std::exception_ptr const& e) noexcept {
        try {
            logger.warn(what, " the ", operation, ": ", exception_message(e));
        } catch (...) {
        }
    };
    {
        std::lock_guard lock(mutex);
        if (completed()) {
            warn("ignoring error raised after completing", error);
            return;
        }
        if (error_slot == nullptr) {
            error_slot = std::move(error);
        }
    }
    cv.notify_all();
    // Also notify the asynchronous waiters, who rethrow when extracting.
    if (auto callback = std::move(finished_callback)) {
        try {
            callback();
        } catch (...) {
            warn("finished callback threw after failing", std::current_exception());
        }
    }
}

}  // namespace rapidsmpf::detail
