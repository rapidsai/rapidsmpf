/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <variant>
#include <vector>

#include <cuda_runtime.h>

#include <rmm/device_buffer.hpp>

#include <rapidsmpf/cuda_event.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/utils.hpp>

namespace rapidsmpf {

class BufferResource;
class MemoryReservation;

/// @brief Enum representing the type of memory.
enum class MemoryType : int {
    DEVICE = 0,  ///< Device memory
    HOST = 1  ///< Host memory
};

/// @brief The lowest memory type that can be spilled to.
constexpr MemoryType LowestSpillType = MemoryType::HOST;

/// @brief Array of all the different memory types.
/// @note Ensure that this array is always sorted in decreasing order of preference.
constexpr std::array<MemoryType, 2> MEMORY_TYPES{{MemoryType::DEVICE, MemoryType::HOST}};

/**
 * @brief Buffer representing device or host memory.
 *
 * @note The constructors are private, use `BufferResource` to construct buffers.
 * @note The memory type (e.g., host or device) is constant and cannot change during
 * the buffer's lifetime.
 * @note A buffer is a stream-ordered object, when passing to a library which is
 * not stream-aware one must ensure that `is_ready` returns `true` otherwise
 * behaviour is undefined.
 */
class Buffer {
    friend class BufferResource;

  public:
    /// @brief  Storage type for the device buffer.
    using DeviceStorageT = std::unique_ptr<rmm::device_buffer>;

    /// @brief  Storage type for the host buffer.
    using HostStorageT = std::unique_ptr<std::vector<uint8_t>>;

    /**
     * @brief  Storage type in Buffer, which could be either host or device memory.
     */
    using StorageT = std::variant<DeviceStorageT, HostStorageT>;

    /**
     * @brief Access the underlying host memory buffer (const).
     *
     * @return A reference to the unique pointer managing the host memory.
     *
     * @throws std::logic_error if the buffer does not manage host memory.
     */
    [[nodiscard]] constexpr HostStorageT const& host() const {
        if (const auto* ref = std::get_if<HostStorageT>(&storage_)) {
            return *ref;
        } else {
            RAPIDSMPF_FAIL("Buffer is not host memory");
        }
    }

    /**
     * @brief Access the underlying device memory buffer (const).
     *
     * @return A const reference to the unique pointer managing the device memory.
     *
     * @throws std::logic_error if the buffer does not manage device memory.
     */
    [[nodiscard]] constexpr DeviceStorageT const& device() const {
        if (const auto* ref = std::get_if<DeviceStorageT>(&storage_)) {
            return *ref;
        } else {
            RAPIDSMPF_FAIL("Buffer is not device memory");
        }
    }

    /**
     * @brief Access the underlying memory buffer (host or device memory).
     *
     * @return A pointer to the underlying host or device memory.
     *
     * @throws std::logic_error if the buffer does not manage any memory.
     */
    [[nodiscard]] std::byte* data();

    /**
     * @brief Access the underlying memory buffer (host or device memory).
     *
     * @return A const pointer to the underlying host or device memory.
     *
     * @throws std::logic_error if the buffer does not manage any memory.
     */
    [[nodiscard]] std::byte const* data() const;

    /**
     * @brief Get the memory type of the buffer.
     *
     * @return The memory type of the buffer.
     *
     * @throws std::logic_error if the buffer is not initialized.
     */
    [[nodiscard]] MemoryType constexpr mem_type() const {
        return std::visit(
            overloaded{
                [](const HostStorageT&) -> MemoryType { return MemoryType::HOST; },
                [](const DeviceStorageT&) -> MemoryType { return MemoryType::DEVICE; }
            },
            storage_
        );
    }

    /**
     * @brief Override the event for the buffer.
     *
     * @note Use this if you want the buffer to sync with an event happening after the
     * original event. Need to be used with care when dealing with multiple streams.
     *
     * @param event The event to set.
     */
    void override_event(std::shared_ptr<CudaEvent> event) {
        event_ = std::move(event);
    }

    /**
     * @brief Get the event for the buffer.
     *
     * @return The event.
     */
    [[nodiscard]] std::shared_ptr<CudaEvent> get_event() const {
        return event_;
    }

    /**
     * @brief Check if the device memory operation has completed.
     *
     * @return true if the device memory operation has completed or no device
     * memory operation was performed, false if it is still in progress.
     */
    [[nodiscard]] bool is_ready() const;

    /**
     * @brief Wait for the device memory operation to complete.
     *
     * @throws rapidsmpf::cuda_error if event wait fails (if set).
     */
    void wait_for_ready() const;

    /**
     * @brief Copy a slice of the buffer to a new buffer allocated from the target
     * reservation.
     *
     * @param offset Non-negative offset from the start of the buffer (in bytes).
     * @param length Length of the slice (in bytes).
     * @param target_reserv Memory reservation for the new buffer.
     * @param stream CUDA stream to use for the copy.
     * @returns A new buffer containing the copied slice.
     */
    [[nodiscard]] std::unique_ptr<Buffer> copy_slice(
        std::ptrdiff_t offset,
        std::size_t length,
        MemoryReservation& target_reserv,
        rmm::cuda_stream_view stream
    ) const;

    /// @brief Delete move and copy constructors and assignment operators.
    Buffer(Buffer&&) = delete;
    Buffer(Buffer const&) = delete;
    Buffer& operator=(Buffer& o) = delete;
    Buffer& operator=(Buffer&& o) = delete;

  private:
    /**
     * @brief Construct a Buffer from host memory.
     *
     * @param host_buffer A unique pointer to a vector containing host memory.
     *
     * @throws std::invalid_argument if `host_buffer` is null.
     */
    Buffer(std::unique_ptr<std::vector<uint8_t>> host_buffer);

    /**
     * @brief Construct a Buffer from device memory.
     *
     * @param device_buffer A unique pointer to a device buffer.
     * @param stream CUDA stream used for the device buffer allocation.
     * @param event The shared event to use for the buffer.
     *
     * @throws std::invalid_argument if `device_buffer` is null.
     * @throws std::invalid_argument if `stream` or `br->mr` isn't the same used by
     * `device_buffer`.
     */
    Buffer(
        std::unique_ptr<rmm::device_buffer> device_buffer,
        rmm::cuda_stream_view stream,
        std::shared_ptr<CudaEvent> event = nullptr
    );

    /**
     * @brief Access the underlying host memory buffer.
     *
     * @return A reference to the unique pointer managing the host memory.
     *
     * @throws std::logic_error if the buffer does not manage host memory.
     */
    [[nodiscard]] constexpr HostStorageT& host() {
        if (auto ref = std::get_if<HostStorageT>(&storage_)) {
            return *ref;
        } else {
            RAPIDSMPF_FAIL("Buffer is not host memory");
        }
    }

    /**
     * @brief Access the underlying device memory buffer.
     *
     * @return A reference to the unique pointer managing the device memory.
     *
     * @throws std::logic_error if the buffer does not manage device memory.
     */
    [[nodiscard]] constexpr DeviceStorageT& device() {
        if (auto ref = std::get_if<DeviceStorageT>(&storage_)) {
            return *ref;
        } else {
            RAPIDSMPF_FAIL("Buffer is not device memory");
        }
    }

    /**
     * @brief Release the underlying device memory buffer.
     *
     * @return The underlying device memory buffer.
     *
     * @throws std::logic_error if the buffer does not manage device memory.
     */
    [[nodiscard]] DeviceStorageT release_device() {
        return std::move(device());
    }

    /**
     * @brief Release the underlying host memory buffer.
     *
     * @return The underlying host memory buffer.
     *
     * @throws std::logic_error if the buffer does not manage host memory.
     */
    [[nodiscard]] HostStorageT release_host() {
        return std::move(host());
    }

  public:
    std::size_t const size;  ///< The size of the buffer in bytes.

  private:
    /// @brief The underlying storage host memory or device memory buffer (where
    /// applicable).
    StorageT storage_;
    /// @brief CUDA event used to track copy operations
    std::shared_ptr<CudaEvent> event_;
};

/**
 * @brief Asynchronously copy data between buffers with optional event tracking.
 *
 * Copies @p size bytes from @p src at @p src_offset into @p dst at @p dst_offset.
 *
 * @param dst Destination buffer.
 * @param src Source buffer.
 * @param size Number of bytes to copy.
 * @param dst_offset Offset (in bytes) into the destination buffer.
 * @param src_offset Offset (in bytes) into the source buffer.
 * @param stream CUDA stream on which to enqueue the copy.
 * @param attach_cuda_event If true, record a CUDA event on @p stream and attach it
 * to the destination buffer to track completion. If false, the caller is responsible
 * for ensuring proper synchronization.
 *
 * @throws std::invalid_argument If out of bounds.
 */
void buffer_copy(
    Buffer& dst,
    Buffer& src,
    std::size_t size,
    std::ptrdiff_t dst_offset,
    std::ptrdiff_t src_offset,
    rmm::cuda_stream_view stream,
    bool attach_cuda_event
);

}  // namespace rapidsmpf
