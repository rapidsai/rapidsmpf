/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <atomic>

#include <cuda_runtime_api.h>

#include <rmm/cuda_stream_view.hpp>

#include <rapidsmpf/error.hpp>

namespace rapidsmpf {

/**
 * @brief Wrapper for cudaEvent_t
 */
class Event {
  public:
    Event();
    ~Event();
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    Event(Event&&) noexcept = delete;
    Event& operator=(Event&&) noexcept = delete;

    /**
     * @brief record the work on `stream` in the event.
     *
     * @param stream The stream to record.
     */
    void record(rmm::cuda_stream_view stream) const;

    /**
     * @brief Get the wrapped event.
     *
     * @return cudaEvent_t the underlying event.
     */
    [[nodiscard]] cudaEvent_t value() const noexcept;

    /**
     * @brief Check if the all the work recorded in the event is complete.
     *
     * @return true if the work is complete.
     */
    [[nodiscard]] bool query() const;

  private:
    mutable std::atomic<bool> done_{false};
    cudaEvent_t event_{nullptr};
};
}  // namespace rapidsmpf
