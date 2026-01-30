/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cstddef>

#include <cuda/memory>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

namespace rapidsmpf::detail {
/**
 * @brief A type-erased uninitialised buffer with an allocation with specified alignment.
 */
struct AlignedBuffer {
    /**
     * @brief Construct the buffer.
     *
     * @param stream Stream for allocations.
     * @param mr Memory resource for allocations.
     * @param size The buffer size.
     * @param alignment The requested alignment.
     */
    explicit AlignedBuffer(
        rmm::cuda_stream_view stream,
        rmm::device_async_resource_ref mr,
        std::size_t size,
        std::size_t alignment
    );
    /**
     * @brief Deallocate the buffer.
     */
    ~AlignedBuffer() noexcept;

    AlignedBuffer(AlignedBuffer const&) = delete;
    AlignedBuffer& operator=(AlignedBuffer const&) = delete;

    /**
     * @brief Move construction.
     *
     * @param other Buffer to be moved from.
     */
    AlignedBuffer(AlignedBuffer&& other) noexcept;

    /**
     * @brief Move assignment.
     *
     * @param other Buffer to be moved from.
     *
     * @throws If `this` buffer already contains an allocation.
     *
     * @return Reference to `this`.
     */
    AlignedBuffer& operator=(AlignedBuffer&& other);

    /**
     * @brief @return The stream the buffer is valid on.
     */
    [[nodiscard]] rmm::cuda_stream_view stream() const noexcept;
    /**
     * @brief @return Pointer to the stored data.
     */
    [[nodiscard]] void* data() noexcept;

    /**
     * @brief @return Size of the buffer in bytes.
     */
    [[nodiscard]] std::size_t size() const noexcept;
    /**
     * @brief @return Memory resource used for allocation/deallocation.
     */
    [[nodiscard]] rmm::device_async_resource_ref mr() noexcept;

  private:
    cuda::mr::any_resource<cuda::mr::device_accessible>
        mr_;  ///< Memory resource for deallocation
    rmm::cuda_stream_view stream_;  ///< Stream we were allocated on
    std::size_t size_;  ///< Size in bytes
    std::size_t alignment_;  ///< Alignment in bytes
    void* data_;  ///< Data
};
}  // namespace rapidsmpf::detail
