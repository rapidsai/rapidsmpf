/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/pausable_thread_loop.hpp>

namespace rapidsmpf::detail {

PausableThreadLoop::~PausableThreadLoop() {
    stop();
}

bool PausableThreadLoop::is_running() const noexcept {
    return !paused_.load(std::memory_order_acquire);
}

void PausableThreadLoop::pause() {
    paused_.store(true, std::memory_order_release);
}

void PausableThreadLoop::resume() {
    paused_.store(false, std::memory_order_release);
    paused_.notify_one();
}

void PausableThreadLoop::stop() {
    active_.store(false, std::memory_order_release);
    paused_.store(false, std::memory_order_release);
    paused_.notify_one();  // Wake up thread to exit
    if (thread_.joinable()) {
        thread_.join();
    }
}

}  // namespace rapidsmpf::detail
