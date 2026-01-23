/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/cuda_event.hpp>

namespace rapidsmpf {

CudaEvent::CudaEvent(unsigned flags) {
    RAPIDSMPF_CUDA_TRY(cudaEventCreateWithFlags(&event_, flags));
}

std::shared_ptr<CudaEvent> CudaEvent::make_shared_record(
    rmm::cuda_stream_view stream, unsigned flags
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
        other.event_ = nullptr;
    }
    return *this;
}

void CudaEvent::record(rmm::cuda_stream_view stream) {
    RAPIDSMPF_CUDA_TRY(cudaEventRecord(event_, stream));
}

[[nodiscard]] bool CudaEvent::CudaEvent::is_ready() const {
    auto result = cudaEventQuery(event_);
    if (result != cudaSuccess && result != cudaErrorNotReady) {
        RAPIDSMPF_CUDA_TRY(result);
    }
    return result == cudaSuccess;
}

void CudaEvent::CudaEvent::host_wait() const {
    RAPIDSMPF_CUDA_TRY(cudaEventSynchronize(event_));
}

void CudaEvent::stream_wait(rmm::cuda_stream_view stream) const {
    RAPIDSMPF_CUDA_TRY(cudaStreamWaitEvent(stream, event_));
}

cudaEvent_t const& CudaEvent::value() const noexcept {
    return event_;
}


}  // namespace rapidsmpf
