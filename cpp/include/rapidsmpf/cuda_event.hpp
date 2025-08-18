/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <atomic>

#include <cuda_runtime.h>

#include <rmm/cuda_stream_view.hpp>

#include <rapidsmpf/error.hpp>

namespace rapidsmpf {


/**
 * @brief RAII wrapper for a CUDA event with convenience methods.
 *
 * Creates a CUDA event on construction and destroys it on destruction.
 *
 * @note To prevent undefined behavior due to unfinished memory operations, events
 * should be used in the following cases, if any of the operations below was performed
 * asynchronously with respect to the host:
 *
 * 1. Before addressing a device buffer's allocation.
 * 2. Before accessing a device buffer's data whose data has been copied from
 *    any location, or that has been processed by a CUDA kernel.
 * 3. Before accessing a host buffer's data whose data has been copied from device,
 *    or processed by a CUDA kernel.
 */
class CudaEvent {
  public:
    /**
     * @brief Construct a CUDA event.
     *
     * @param flags CUDA event creation flags.
     *
     * @throws rapidsmpf::cuda_error if cudaEventCreateWithFlags fails.
     */
    CudaEvent(unsigned flags = cudaEventDisableTiming);

    /**
     * @brief Destroy the CUDA event.
     *
     * Automatically releases the underlying CUDA event resource.
     */
    ~CudaEvent() noexcept;

    CudaEvent(CudaEvent const&) = delete;  ///< Non-copyable.
    CudaEvent& operator=(CudaEvent const&) = delete;  ///< Non-copy-assignable.

    /**
     * @brief Move constructor.
     *
     * @param other Source CudaEvent to move from.
     */
    CudaEvent(CudaEvent&& other) noexcept;

    /**
     * @brief Move assignment operator.
     *
     * @param other Source CudaEvent to move from.
     * @return Reference to this object.
     */
    CudaEvent& operator=(CudaEvent&& other);

    /**
     * @brief Record the event on a CUDA stream.
     *
     * Marks the event as occurring after all prior operations on the given stream.
     *
     * @param stream The CUDA stream to record the event on.
     *
     * @throws rapidsmpf::cuda_error if cudaEventRecord fails.
     */
    void record(rmm::cuda_stream_view stream);

    /**
     * @brief Check if the CUDA event has been completed.
     *
     * @return true if the event has been completed, false otherwise.
     *
     * @throws rapidsmpf::cuda_error if cudaEventQuery fails.
     */
    [[nodiscard]] bool is_ready();

    /**
     * @brief Wait for the event to be completed (blocking).
     *
     * @throws rapidsmpf::cuda_error if cudaEventSynchronize fails.
     */
    void wait();

    /**
     * @brief Access the underlying CUDA event handle.
     *
     * @return Const reference to the underlying cudaEvent_t.
     */
    [[nodiscard]] cudaEvent_t const& value() const noexcept;

    /**
     * @brief Implicit conversion operator to the CUDA event handle.
     *
     * @return Const reference to the underlying cudaEvent_t.
     */
    operator cudaEvent_t const&() const noexcept {
        return value();
    }

  private:
    cudaEvent_t event_{};  ///< Underlying CUDA event handle.
    std::atomic<bool> done_{false};  ///< Cache of the event status.
};


}  // namespace rapidsmpf
