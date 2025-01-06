/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include <limits>
#include <mutex>
#include <unordered_map>

#include <rapidsmp/buffer/buffer.hpp>
#include <rapidsmp/error.hpp>

#include "rapidsmp/utils.hpp"

namespace rapidsmp {

/**
 * @brief Represents a reservation for future memory allocation.
 *
 * A reservation is returned by `BufferResource::reserve` and must be used when allocating
 * buffers through the `BufferResource`.
 */
class MemoryReservation {
    friend class BufferResource;

  public:
    /**
     * @brief Destructor for the memory reservation.
     *
     * Cleans up resources associated with the reservation.
     */
    ~MemoryReservation() noexcept;

    /// @brief A memory reservation is moveable.
    MemoryReservation(MemoryReservation&& o)
        : MemoryReservation{o.mem_type_, o.br_, std::exchange(o.size_, 0)} {}

    /// @brief A memory reservation is moveable.
    MemoryReservation& operator=(MemoryReservation&& o) noexcept {
        mem_type_ = o.mem_type_;
        br_ = o.br_;
        size_ = std::exchange(o.size_, 0);
        return *this;
    }

    /// @brief A memory reservation is not copyable.
    MemoryReservation(MemoryReservation const&) = delete;
    MemoryReservation& operator=(MemoryReservation const&) = delete;

    /**
     * @brief Get the remaining size of the reserved memory.
     *
     * @return The size of the reserved memory in bytes.
     */
    [[nodiscard]] std::size_t size() const {
        return size_;
    }

  private:
    /**
     * @brief Constructs a memory reservation.
     *
     * @param mem_type The type of memory associated with this reservation.
     * @param br Pointer to the buffer resource managing this reservation.
     * @param size The size of the reserved memory in bytes.
     */
    constexpr MemoryReservation(MemoryType mem_type, BufferResource* br, std::size_t size)
        : mem_type_{mem_type}, br_{br}, size_{size} {}

  private:
    MemoryType mem_type_;  ///< The type of memory for this reservation.
    BufferResource* br_;  ///< The buffer resource that manages this reservation.
    std::size_t size_;  ///< The remaining size of the reserved memory in bytes.
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
     * @brief
     *
     * Doesn't have to be thread-safe but cannot lock any resource that might reserve
     * memory using this spill resource.
     *
     */
    using MemoryAvailable = std::function<std::int64_t()>;

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
        std::unordered_map<MemoryType, MemoryAvailable> memory_available =
            {{MemoryType::DEVICE, std::numeric_limits<std::int64_t>::max},
             {MemoryType::HOST, std::numeric_limits<std::int64_t>::max}}
    )
        : device_mr_{device_mr}, memory_available_{std::move(memory_available)} {}

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
     * @brief Reserve an amount of the specified memory type.
     *
     * Creates a new reservation to inform about upcoming buffer allocations. E.g.,
     * the `Shuffler::extract()` use this to reserve device memory for the next
     * output pertition before moving each individual chunk to device memory.
     *
     * Additionally, the amount of "overbooking" is also returned.
     *
     * @param mem_type The target memory type.
     * @param size The number of bytes to reserve.
     * @return An allocation reservation and the amount of "overbooking". Or null
     */
    virtual std::pair<MemoryReservation, std::size_t> reserve(
        MemoryType mem_type, size_t size, bool allow_overbooking
    );

    /**
     * @brief Consume a portion of the reserved memory.
     *
     * Reduces the remaining size of the reserved memory by the specified amount.
     *
     * @param target The memory type of the reservation.
     * @param size The size to consume in bytes.
     * @return The remaining size of the reserved memory after consumption.
     *
     * @throws std::invalid_argument if the memory type does not match the reservation.
     * @throws std::overflow_error if the requested size exceeds the reserved memory size.
     */
    virtual std::size_t release(
        MemoryReservation& reservation, MemoryType target, std::size_t size
    );

    /**
     * @brief Allocate a buffer of the specified memory type.
     *
     * @param mem_type The target memory type (host or device).
     * @param size The size of the buffer in bytes.
     * @param stream CUDA stream to use for device allocations.
     * @param reservation The reservation to use for memory allocations.
     * @return A unique pointer to the allocated Buffer.
     *
     * @throws std::invalid_argument if the memory type does not match the reservation.
     * @throws std::overflow_error if the reservation isn't big enough.
     */
    virtual std::unique_ptr<Buffer> allocate(
        MemoryType mem_type,
        std::size_t size,
        rmm::cuda_stream_view stream,
        MemoryReservation& reservation
    );

    /**
     * @brief Allocate a new buffer.
     *
     * The base implementation first try to reserve device memory then host memory.
     *
     * @param size The size of the buffer in bytes.
     * @param stream CUDA stream to use for device allocations.
     * @return A unique pointer to the allocated Buffer.
     */
    virtual std::unique_ptr<Buffer> allocate(
        std::size_t size, rmm::cuda_stream_view stream
    );

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
        MemoryReservation& reservation
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
        MemoryReservation& reservation
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
        MemoryReservation& reservation
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
        MemoryReservation& reservation
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
    /**
     * @brief Factory method to create a new memory reservation.
     *
     * This is needed because the ctor of `MemoryReservation` is private.
     *
     * @param mem_type The type of memory associated with this reservation.
     * @param br Pointer to the buffer resource managing this reservation.
     * @param size The size of the reserved memory in bytes.
     */
    static MemoryReservation create_memory_reservation(
        MemoryType mem_type, BufferResource* br, std::size_t size
    ) {
        return MemoryReservation{mem_type, br, size};
    }

    static std::size_t release_memory_reservation(
        MemoryReservation& reservation, MemoryType target, std::size_t size
    ) {
        RAPIDSMP_EXPECTS(
            reservation.mem_type_ == target,
            "the memory type of MemoryReservation doesn't match",
            std::invalid_argument
        );
        RAPIDSMP_EXPECTS(
            size <= reservation.size_,
            "MemoryReservation(" + format_nbytes(reservation.size_)
                + ") isn't big enough (" + format_nbytes(size) + ")",
            std::overflow_error
        );
        return reservation.size_ -= size;
    }

  protected:
    std::mutex mutex_;  ///< For thread-safe access to memory reservations.
    rmm::device_async_resource_ref device_mr_;  ///< RMM device memory resource reference.
    std::unordered_map<MemoryType, MemoryAvailable> const memory_available_;
    std::unordered_map<MemoryType, std::size_t> memory_reserved_{
        {MemoryType::DEVICE, 0}, {MemoryType::HOST, 0}
    };
};

}  // namespace rapidsmp
