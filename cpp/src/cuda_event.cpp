/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>

#include <cuda_runtime_api.h>

#include <rmm/cuda_stream_view.hpp>

#include <rapidsmpf/cuda_event.hpp>

namespace rapidsmpf {

Event::Event() {
    RAPIDSMPF_CUDA_TRY(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
}

Event::~Event() {
    cudaEventDestroy(event_);
}

void Event::record(rmm::cuda_stream_view stream) const {
    if (!query()) {
        RAPIDSMPF_CUDA_TRY(cudaEventRecord(event_, stream.value()));
    }
}

cudaEvent_t Event::value() const noexcept {
    return event_;
}

[[nodiscard]] bool Event::query() const {
    if (!done_.load(std::memory_order_relaxed)) {
        auto result = cudaEventQuery(event_);
        if (result == cudaSuccess) {
            done_.store(true, std::memory_order_relaxed);
            return true;
        } else if (result != cudaErrorNotReady) {
            RAPIDSMPF_CUDA_TRY(result);
        }
    }
    return done_.load(std::memory_order_relaxed);
}
}  // namespace rapidsmpf
