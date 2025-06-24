/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include <thread>

#include <rapidsmpf/locking.hpp>

namespace rapidsmpf {

detail::timeout_lock_guard::timeout_lock_guard(
    std::timed_mutex& mutex,
    char const* filename,
    int line_number,
    Duration const& timeout
)
    : lock_(mutex, std::defer_lock) {
    while (!lock_.try_lock_for(timeout)) {
        std::stringstream ss;
        ss << "[possible deadlock] thread 0x" << std::hex << std::this_thread::get_id()
           << " blocking (" << std::dec << timeout.count() << " s) at " << filename << ":"
           << line_number;
        std::cerr << ss.str() << std::endl;
    }
}


}  // namespace rapidsmpf
