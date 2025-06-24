/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include <map>
#include <thread>

#include <rapidsmpf/locking.hpp>

namespace rapidsmpf::detail {

static thread_local std::map<uintptr_t, std::tuple<char const*, int>> held_locks{};

timeout_lock_guard::timeout_lock_guard(
    std::timed_mutex& mutex,
    char const* filename,
    int line_number,
    Duration const& timeout
)
    : lock_(mutex, std::defer_lock), mid_{reinterpret_cast<uintptr_t>(&mutex)} {
    while (!lock_.try_lock_for(timeout)) {
        std::stringstream ss;
        ss << "[possible deadlock] thread 0x" << std::hex << std::this_thread::get_id()
           << " blocking (" << std::dec << timeout.count() << " s) at " << filename << ":"
           << line_number;
        ss << "\n  thread is trying to acquire mutex 0x" << std::hex << mid_ << std::dec;
        ss << "\n  thread holds the following mutexes already:\n";
        for (auto& [mutex_id, info] : held_locks) {
            auto& [filename, line_number] = info;
            ss << "    " << std::hex << "mutex 0x" << mutex_id << " at " << filename
               << ":" << std::dec << line_number << "\n";
        }
        std::cerr << ss.str() << std::endl;
    }
    held_locks[mid_] = {filename, line_number};
}

timeout_lock_guard::~timeout_lock_guard() {
    held_locks.erase(mid_);
}

}  // namespace rapidsmpf::detail
