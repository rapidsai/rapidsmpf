/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <mutex>

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
 * should be used in the following cases if any of the operations below were performed
 * asynchronously with respect to the host:
 *   1. Before addressing a device buffer's allocation.
 *   2. Before accessing a device buffer's data that has been copied from
 *      any location, or processed by a CUDA kernel.
 *   3. Before accessing a host buffer's data that has been copied from device
 *      or processed by a CUDA kernel.
 *
 * @note `CudaEvent` objects must not have static storage duration, since CUDA resources
 * are not guaranteed to be valid during program initialization or shutdown.
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
     * @brief Create and record a CUDA event on a given stream.
     *
     * Convenience factory that constructs a shared `CudaEvent` with the specified
     * creation flags, immediately records it on the provided stream, and returns
     * it as a `std::shared_ptr`.
     *
     * @param stream CUDA stream on which to record the event.
     * @param flags CUDA event creation flags.
     * @return A shared pointer to the newly created and recorded CudaEvent.
     *
     * @throws rapidsmpf::cuda_error if event creation or recording fails.
     */
    static std::shared_ptr<CudaEvent> make_shared_record(
        rmm::cuda_stream_view stream, unsigned flags = cudaEventDisableTiming
    );

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
    [[nodiscard]] bool is_ready() const;

    /**
     * @brief Wait for the event to be completed (blocking).
     *
     * @throws rapidsmpf::cuda_error if cudaEventSynchronize fails.
     */
    void host_wait() const;

    /**
     * @brief Make a CUDA stream wait on this event (non-blocking).
     *
     * Ensures that all operations submitted to the given stream after this call
     * will not begin execution until this event has completed.
     *
     * @param stream CUDA stream that should wait for the event.
     *
     * @throws rapidsmpf::cuda_error if cudaStreamWaitEvent fails.
     */
    void stream_wait(rmm::cuda_stream_view stream) const;

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
    cudaEvent_t event_{};
};


}  // namespace rapidsmpf
