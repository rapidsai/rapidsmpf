/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <mutex>

#include <rapidsmp/buffer/buffer.hpp>
#include <rapidsmp/error.hpp>

#include "rapidsmp/utils.hpp"

namespace rapidsmp {

class MemoryReservation {
  public:
    constexpr MemoryReservation(
        MemoryType mem_type, BufferResource* const br, std::size_t size
    )
        : mem_type{mem_type}, br{br}, size_{size} {}

    ~MemoryReservation() noexcept;

    /// @brief An reservation isn't moveable or copyable.
    MemoryReservation(MemoryReservation&& other) = delete;
    MemoryReservation& operator=(MemoryReservation&&) = delete;
    MemoryReservation(MemoryReservation const&) = delete;
    MemoryReservation& operator=(MemoryReservation const&) = delete;

    [[nodiscard]] std::size_t size() const {
        return size_;
    }

    std::size_t use(MemoryType target, std::size_t size) {
        RAPIDSMP_EXPECTS(
            mem_type == target,
            "the memory type of MemoryReservation doesn't match",
            std::invalid_argument
        );
        std::lock_guard const lock(mutex_);
        RAPIDSMP_EXPECTS(
            size <= size_,
            "MemoryReservation(" + format_nbytes(size_) + ") isn't big enough ("
                + format_nbytes(size) + ")",
            std::overflow_error
        );
        return size_ -= size;
    }

  public:
    MemoryType const mem_type;
    BufferResource* const br;

  private:
    std::size_t size_;
    std::mutex mutex_;
};

/**
 * @brief Base class for managing buffer resources.
 *
 * This class handles memory allocation and transfers between different memory types
 * (e.g., host and device). All memory operations in rapidsmp, such as those performed
 * by the Shuffler, rely on a buffer resource for memory management.
 *
 * This base class always use the highest memory type in the provided memory hierarchy.
 *
 * An alternative allocation strategic can be implemented in a derived class by overriding
 * one or more of the virtual methods.
 *
 * @note Similar to RMM's memory resource, the `BufferResource` instance must outlive all
 * allocated buffers and allocation tokens.
 */
class BufferResource {
  public:
    /**
     * @brief Constructs a buffer resource that uses a memory hierarchy to decide the
     * memory type of each new allocation.
     *
     * @param device_mr Reference to the RMM device memory resource used for all device
     * allocations, which must outlive `BufferResource` and all the created buffers.
     * @param memory_hierarchy The memory hierarchy to use (the base class always uses
     * the highest memory type).
     */
    BufferResource(
        rmm::device_async_resource_ref device_mr,
        MemoryHierarchy memory_hierarchy = {MemoryType::DEVICE, MemoryType::HOST}
    )
        : device_mr_{device_mr}, memory_hierarchy_{memory_hierarchy} {}

    virtual ~BufferResource() noexcept = default;

    /**
     * @brief Get the RMM device memory resource.
     *
     * @return Reference to the RMM resource used for device allocations.
     */
    [[nodiscard]] rmm::device_async_resource_ref device_mr() const noexcept {
        return device_mr_;
    }

    /**
     * @brief Get the memory hierarchy.
     *
     * @return Reference to an array of memory types ordered by the hierarchy.
     */
    [[nodiscard]] MemoryHierarchy const& memory_hierarchy() const noexcept {
        return memory_hierarchy_;
    }

    /**
     * @brief Reserve an amount of the specified memory type.
     *
     * This can be used by derived classes to help determinate the memory type of upcoming
     * buffer allocations. E.g., the Shuffler could use this to reserve device memory for
     * the next output pertition before moving each individual chunk to device memory.
     *
     * @param mem_type The target memory type.
     * @param size The number of bytes to reserve.
     * @return An allocation token and the amount of "overbooking".
     */
    virtual std::pair<std::unique_ptr<MemoryReservation>, std::size_t> reserve(
        MemoryType mem_type, size_t size
    );

    virtual void release(MemoryReservation const& token) noexcept;

    /**
     * @brief Allocate a buffer of the specified memory type.
     *
     * @param mem_type The target memory type (host or device).
     * @param size The size of the buffer in bytes.
     * @param stream CUDA stream to use for device allocations.
     * @return A unique pointer to the allocated Buffer.
     */
    virtual std::unique_ptr<Buffer> allocate(
        MemoryType mem_type,
        size_t size,
        rmm::cuda_stream_view stream,
        std::unique_ptr<MemoryReservation>& token
    );

    /**
     * @brief Allocate a new buffer.
     *
     * The base implementation always use the highest memory type in the memory
     * hierarchy.
     *
     * @param size The size of the buffer in bytes.
     * @param stream CUDA stream to use for device allocations.
     * @return A unique pointer to the allocated Buffer.
     */
    virtual std::unique_ptr<Buffer> allocate(size_t size, rmm::cuda_stream_view stream) {
        auto mem_type = memory_hierarchy().at(0);  // Always pick the highest memory type.
        auto [token, overbooking] = reserve(mem_type, size);
        // TODO: check overbooking, do we need to spill?
        return allocate(mem_type, size, stream, token);
    }

    /**
     * @brief Move host vector data into a Buffer.
     *
     * @param data A unique pointer to the vector containing host data.
     * @param stream CUDA stream for any necessary operations.
     * @return A unique pointer to the resulting Buffer.
     */
    virtual std::unique_ptr<Buffer> move(
        std::unique_ptr<std::vector<uint8_t>> data, rmm::cuda_stream_view stream
    );

    /**
     * @brief Move device buffer data into a Buffer.
     *
     * @param data A unique pointer to the device buffer.
     * @param stream CUDA stream for any necessary operations.
     * @return A unique pointer to the resulting Buffer.
     */
    virtual std::unique_ptr<Buffer> move(
        std::unique_ptr<rmm::device_buffer> data, rmm::cuda_stream_view stream
    );

    /**
     * @brief Move a Buffer to the specified memory type.
     *
     * If and only if moving between different memory types will this perform a copy.
     *
     * @param target The target memory type.
     * @param buffer The buffer to move.
     * @param stream CUDA stream for the operation.
     * @return A unique pointer to the moved Buffer.
     */
    virtual std::unique_ptr<Buffer> move(
        MemoryType target,
        std::unique_ptr<Buffer> buffer,
        rmm::cuda_stream_view stream,
        std::unique_ptr<MemoryReservation>& token
    );

    /**
     * @brief Move a Buffer to a device buffer.
     *
     * If and only if moving between different memory types will this perform a copy.
     *
     * @param buffer The buffer to move.
     * @param stream CUDA stream for the operation.
     * @return A unique pointer to the resulting device buffer.
     */
    virtual std::unique_ptr<rmm::device_buffer> move_to_device_buffer(
        std::unique_ptr<Buffer> buffer,
        rmm::cuda_stream_view stream,
        std::unique_ptr<MemoryReservation>& token
    );

    /**
     * @brief Move a Buffer to a host vector.
     *
     * If and only if moving between different memory types will this perform a copy.
     *
     * @param buffer The buffer to move.
     * @param stream CUDA stream for the operation.
     * @return A unique pointer to the resulting host vector.
     */
    virtual std::unique_ptr<std::vector<uint8_t>> move_to_host_vector(
        std::unique_ptr<Buffer> buffer,
        rmm::cuda_stream_view stream,
        std::unique_ptr<MemoryReservation>& token
    );

    /**
     * @brief Create a copy of a Buffer in the specified memory type.
     *
     * Unlike `move()`, this always performs a copy operation.
     *
     * @param target The target memory type.
     * @param buffer The buffer to copy.
     * @param stream CUDA stream for the operation.
     * @return A unique pointer to the new Buffer.
     */
    virtual std::unique_ptr<Buffer> copy(
        MemoryType target,
        std::unique_ptr<Buffer> const& buffer,
        rmm::cuda_stream_view stream,
        std::unique_ptr<MemoryReservation>& token
    );

    /**
     * @brief This finalizer is called when a buffer is being destructured.
     *
     * The base implementation does nothing.
     *
     * @note This is only called on initialized buffers i.e. only if
     * `buffer->is_moved() == false`.
     *
     * @param buffer The buffer being destructured.
     */
    virtual void finalizer(Buffer* const buffer) noexcept {}

  protected:
    rmm::device_async_resource_ref device_mr_;  ///< RMM device memory resource reference.
    MemoryHierarchy memory_hierarchy_;  ///< The hierarchy of memory types to use.
};

}  // namespace rapidsmp
