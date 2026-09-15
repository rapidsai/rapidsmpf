/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <utility>

#include <cuda/stream>

#include <rapidsmpf/cuda_event.hpp>

namespace rapidsmpf {

CudaEvent::CudaEvent(unsigned flags) {
    RAPIDSMPF_CUDA_TRY(cudaEventCreateWithFlags(&event_, flags));
}

std::shared_ptr<CudaEvent> CudaEvent::make_shared_record(
    cuda::stream_ref stream, unsigned flags
) {
    auto ret = std::make_shared<CudaEvent>(flags);
    ret->record(stream);
    return ret;
}

CudaEvent::~CudaEvent() noexcept {
    if (event_ != nullptr) {
        cudaEventDestroy(event_);
    }
}

CudaEvent::CudaEvent(CudaEvent&& other) noexcept : event_{other.event_} {
    other.event_ = nullptr;
}

CudaEvent& CudaEvent::operator=(CudaEvent&& other) {
    if (this != &other) {
        RAPIDSMPF_EXPECTS(
            event_ == nullptr,
            "cannot move into an already-initialized CudaEvent",
            std::invalid_argument
        );
        event_ = std::exchange(other.event_, nullptr);
    }
    return *this;
}

void CudaEvent::record(cuda::stream_ref stream) {
    RAPIDSMPF_CUDA_TRY(cudaEventRecord(event_, stream.get()));
}

[[nodiscard]] bool CudaEvent::is_ready() const {
    auto result = cudaEventQuery(event_);
    if (result != cudaSuccess && result != cudaErrorNotReady) {
        RAPIDSMPF_CUDA_TRY(result);
    }
    return result == cudaSuccess;
}

void CudaEvent::host_wait() const {
    RAPIDSMPF_CUDA_TRY(cudaEventSynchronize(event_));
}

void CudaEvent::stream_wait(cuda::stream_ref stream) const {
    RAPIDSMPF_CUDA_TRY(cudaStreamWaitEvent(stream.get(), event_));
}

cudaEvent_t const& CudaEvent::value() const noexcept {
    return event_;
}


}  // namespace rapidsmpf
