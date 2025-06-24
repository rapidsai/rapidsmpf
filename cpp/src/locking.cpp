/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include <thread>

#include <rapidsmpf/locking.hpp>

namespace rapidsmpf::detail {

// Define the thread-local static variable
// thread_local std::vector<std::pair<char const*, int>> timeout_lock_guard::held_locks_;

timeout_lock_guard::timeout_lock_guard(
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
        ss << "\n  thread holds the following mutexes already:\n";
        for (auto& [mutex_id, filename, line_number] : held_locks_) {
            ss << "    " << std::hex << "mutex 0x" << mutex_id << " at " << filename
               << ":" << std::dec << line_number << "\n";
        }
        std::cerr << ss.str() << std::endl;
    }
    held_locks_.emplace_back(reinterpret_cast<uintptr_t>(&mutex), filename, line_number);
}

timeout_lock_guard::~timeout_lock_guard() {
    if (!held_locks_.empty()) {
        held_locks_.pop_back();
    }
}

}  // namespace rapidsmpf::detail
