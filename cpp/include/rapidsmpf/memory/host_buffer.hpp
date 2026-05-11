/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <span>
#include <vector>

#include <cuda/memory_resource>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/host_memory_resource.hpp>

namespace rapidsmpf {


/**
 * @brief Block of host memory.
 */
class HostBuffer {
  public:
    /**
     * @brief Allocate a new host buffer.
     *
     * If `size` is greater than zero, memory is allocated using the provided memory
     * resource and stream. If `size` is zero, the buffer is created empty.
     *
     * @param size Number of bytes to allocate.
     * @param stream CUDA stream on which allocation and deallocation occur.
     * @param mr Host-accessible memory resource used for allocation. Taken by value
     * so the buffer shares ownership of the resource (e.g. bumps the refcount
     * when constructed from a shared-ownership resource); an implicit
     * conversion from `rmm::host_async_resource_ref` is also supported.
     */
    HostBuffer(
        std::size_t size,
        rmm::cuda_stream_view stream,
        cuda::mr::any_resource<cuda::mr::host_accessible> mr
    );

    ~HostBuffer() noexcept;

    // Non copyable
    HostBuffer(HostBuffer const&) = delete;
    HostBuffer& operator=(HostBuffer const&) = delete;

    /**
     * @brief Move constructor.
     *
     * Transfers ownership of the underlying memory. The moved-from object
     * becomes empty.
     *
     * @param other The buffer to move from.
     */
    HostBuffer(HostBuffer&& other) noexcept;

    /**
     * @brief Move assignment operator.
     *
     * Transfers ownership of the underlying memory. The current buffer must be
     * empty before assignment.
     *
     * @param other The buffer to move from.
     * @return Reference to this object.
     *
     * @throws std::invalid_argument if this buffer is already initialized.
     */
    HostBuffer& operator=(HostBuffer&& other);

    /**
     * @brief Stream-ordered deallocates the buffer, if allocated.
     *
     * After deallocation the buffer becomes empty. It is safe to call this
     * method multiple times.
     */
    void deallocate_async() noexcept;

    /**
     * @brief Get the CUDA stream associated with this buffer.
     *
     * @return CUDA stream view.
     */
    [[nodiscard]] rmm::cuda_stream_view stream() const noexcept;

    /**
     * @brief Get the size of the buffer in bytes.
     *
     * @return Number of bytes in the buffer.
     */
    [[nodiscard]] std::size_t size() const noexcept;

    /**
     * @brief Check whether the buffer is empty.
     *
     * @return True if no memory is allocated.
     */
    [[nodiscard]] bool empty() const noexcept;

    /**
     * @brief Get a pointer to the buffer data.
     *
     * @return Pointer to the underlying memory.
     */
    [[nodiscard]] std::byte* data() noexcept;

    /**
     * @brief Get a const pointer to the buffer data.
     *
     * @return Const pointer to the underlying memory.
     */
    [[nodiscard]] std::byte const* data() const noexcept;

    /**
     * @brief Copy the contents of the buffer into a host `std::vector`.
     *
     * This is primarily intended for debugging or testing. It performs a
     * stream synchronization before and after the copy.
     *
     * @return A vector containing the copied bytes.
     */
    [[nodiscard]] std::vector<std::uint8_t> copy_to_uint8_vector() const;

    /**
     * @brief Set the associated CUDA stream.
     *
     * This function only updates the buffer's associated CUDA stream. It does not
     * perform any stream synchronization or establish ordering guarantees between
     * the previous stream and @p stream. Callers must ensure that any work
     * previously enqueued on the old stream that may access the buffer has completed
     * or is otherwise properly synchronized before switching streams.
     *
     * @param stream The new CUDA stream.
     */
    void set_stream(rmm::cuda_stream_view stream);

    /**
     * @brief Construct a `HostBuffer` by copying data from a `std::vector<std::uint8_t>`.
     *
     * A new buffer is allocated using the provided memory resource and stream.
     * This helper is intended for tests and small debug utilities.
     *
     * @param data Source vector containing the bytes to copy.
     * @param stream CUDA stream used for allocation and copy.
     * @param mr Host memory resource used to allocate the buffer.
     *
     * @return A new `HostBuffer` containing a copy of `data`.
     */
    static HostBuffer from_uint8_vector(
        std::vector<std::uint8_t> const& data,
        rmm::cuda_stream_view stream,
        rmm::host_async_resource_ref mr
    );

    /**
     * @brief Construct a `HostBuffer` by taking ownership of a
     * `std::vector<std::uint8_t>`.
     *
     * The buffer takes ownership of the vector's memory. The vector is moved into
     * internal storage and will be destroyed when the `HostBuffer` is destroyed.
     *
     * @param data Vector to take ownership of (will be moved).
     * @param stream CUDA stream to associate with this buffer.
     *
     * @return A new `HostBuffer` owning the vector's memory.
     */
    static HostBuffer from_owned_vector(
        std::vector<std::uint8_t>&& data, rmm::cuda_stream_view stream
    );

    /**
     * @brief Construct a `HostBuffer` by taking ownership of an `rmm::device_buffer`.
     *
     * The buffer takes ownership of the device buffer. The caller must ensure that
     * the device buffer contains host-accessible memory (e.g., pinned host memory
     * allocated via a managed or pinned memory resource).
     *
     * @warning The caller is responsible for ensuring the device buffer's memory is
     * host-accessible. Using this with non-host-accessible device memory will result
     * in a std::invalid_argument exception.
     *
     * @param pinned_host_buffer Device buffer to take ownership of.
     * @param stream CUDA stream to associate with this buffer.
     *
     * @return A new `HostBuffer` owning the device buffer's memory.
     *
     * @throws std::invalid_argument if @p pinned_host_buffer is null.
     * @throws std::logic_error if @p pinned_host_buffer is not of pinned memory host
     * memory type (see warning for details).
     *
     * @warning The caller is responsible to ensure @p pinned_host_buffer is of pinned
     * host memory type. If non-pinned host buffer leads to an irrecoverable condition.
     */
    static HostBuffer from_rmm_device_buffer(
        std::unique_ptr<rmm::device_buffer> pinned_host_buffer,
        rmm::cuda_stream_view stream
    );

  private:
    /**
     * @brief Private constructor for creating a buffer with owned storage.
     *
     * @param span View of the owned memory.
     * @param stream CUDA stream associated with this buffer.
     * @param deallocate_fn Callable invoked with the current stream to release the
     * underlying memory. It captures all resources needed for deallocation (e.g.
     * memory resource, raw pointer, size).
     */
    HostBuffer(
        std::span<std::byte> span,
        rmm::cuda_stream_view stream,
        std::function<void(rmm::cuda_stream_view)> deallocate_fn
    );

    rmm::cuda_stream_view stream_;
    std::span<std::byte> span_{};
    /// @brief Callable that releases the underlying memory when invoked with the current
    /// stream. Null when the buffer is empty.
    std::function<void(rmm::cuda_stream_view)> deallocate_fn_{};
};

}  // namespace rapidsmpf
