/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include <map>
#include <thread>

#include <rapidsmpf/locking.hpp>

namespace rapidsmpf::detail {
namespace {

using held_locks_t = std::map<uintptr_t, std::tuple<char const*, int>>;

std::mutex held_locks_by_thread_mutex;
std::map<std::thread::id, std::unique_ptr<held_locks_t>> held_locks_by_thread{};

held_locks_t& get_locks_held_by_thread() {
    std::lock_guard<std::mutex> const lock(held_locks_by_thread_mutex);
    auto& held_ptr = held_locks_by_thread[std::this_thread::get_id()];
    if (!held_ptr) {
        held_ptr = std::make_unique<held_locks_t>();
    }
    return *held_ptr;
}

};  // namespace

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
        ss << ", the following is all known acquired mutexes:\n";
        std::lock_guard<std::mutex> const lock(held_locks_by_thread_mutex);
        for (auto& [thread_id, held_locks] : held_locks_by_thread) {
            for (auto& [mutex_id, info] : *held_locks) {
                auto& [filename, line_number] = info;
                ss << "    " << std::hex << "thread 0x" << thread_id
                   << " acquired mutex 0x" << mutex_id << " at " << filename << ":"
                   << std::dec << line_number << "\n";
            }
        }
        std::cerr << ss.str() << std::endl;
    }
    get_locks_held_by_thread()[mid_] = {filename, line_number};
}

timeout_lock_guard::~timeout_lock_guard() {
    get_locks_held_by_thread().erase(mid_);
}

}  // namespace rapidsmpf::detail
