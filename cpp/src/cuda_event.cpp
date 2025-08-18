/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/cuda_event.hpp>

namespace rapidsmpf {

CudaEvent::CudaEvent(unsigned flags) {
    RAPIDSMPF_CUDA_TRY(cudaEventCreateWithFlags(&event_, flags));
}

CudaEvent::~CudaEvent() noexcept {
    cudaEventDestroy(event_);
}

CudaEvent::CudaEvent(CudaEvent&& other) noexcept
    : event_{other.event_}, done_{other.done_.load()} {
    other.event_ = nullptr;
}

CudaEvent& CudaEvent::operator=(CudaEvent&& other) {
    RAPIDSMPF_EXPECTS(
        event_ == nullptr, "cannot move into a non-empty CudaEvent", std::invalid_argument
    );
    if (this != &other) {
        event_ = other.event_;
        done_.store(other.done_.load());
        other.event_ = nullptr;
    }
    return *this;
}

void CudaEvent::record(rmm::cuda_stream_view stream) {
    RAPIDSMPF_CUDA_TRY(cudaEventRecord(event_, stream));
}

[[nodiscard]] bool CudaEvent::CudaEvent::is_ready() {
    if (!done_.load(std::memory_order_acquire)) {
        auto result = cudaEventQuery(event_);
        done_.store(result == cudaSuccess, std::memory_order_release);
        if (result != cudaSuccess && result != cudaErrorNotReady) {
            RAPIDSMPF_CUDA_TRY(result);
        }
        return result == cudaSuccess;
    }
    return true;
}

void CudaEvent::CudaEvent::wait() {
    if (!done_.load(std::memory_order_relaxed)) {
        RAPIDSMPF_CUDA_TRY(cudaEventSynchronize(event_));
        done_.store(true, std::memory_order_relaxed);
    }
}

cudaEvent_t const& CudaEvent::value() const noexcept {
    return event_;
}


}  // namespace rapidsmpf
