/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <array>
#include <atomic>
#include <memory>
#include <variant>
#include <vector>

#include <cuda_runtime.h>

#include <rmm/device_buffer.hpp>

#include <rapidsmpf/error.hpp>

namespace rapidsmpf {

class BufferResource;
class Event;
class MemoryReservation;

/// @brief Enum representing the type of memory.
enum class MemoryType : int {
    DEVICE = 0,  ///< Device memory
    HOST = 1  ///< Host memory
};

/// @brief Array of all the different memory types.
constexpr std::array<MemoryType, 2> MEMORY_TYPES{{MemoryType::DEVICE, MemoryType::HOST}};

namespace {
/// @brief Helper for overloaded lambdas using std::visit.
template <class... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};
/// @brief Explicit deduction guide
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

}  // namespace

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
    /**
     * @brief CUDA event to provide synchronization among set of chunks.
     *
     * This event is used to serve as a synchronization point for a set of chunks
     * given a user-specified stream.
     *
     * @note To prevent undefined behavior due to unfinished memory operations, events
     * should be used in the following cases, if any of the operations below was
     * performed *asynchronously with respect to the host*:
     * 1. Before addressing a device buffer's allocation.
     * 2. Before accessing a device buffer's data whose data has been copied from
     * any location, or that has been processed by a CUDA kernel.
     * 3. Before accessing a host buffer's data whose data has been copied from device,
     * or processed by a CUDA kernel.
     */
    class Event {
      public:
        /**
         * @brief Construct a CUDA event for a given stream.
         *
         * @param stream CUDA stream used for device memory operations
         */
        Event(rmm::cuda_stream_view stream);

        /**
         * @brief Destructor for Event.
         *
         * Cleans up the CUDA event if one was created.
         */
        ~Event();

        /**
         * @brief Check if the CUDA event has been completed.
         *
         * @return true if the event has been completed, false otherwise.
         *
         * @throws rapidsmpf::cuda_error if cudaEventQuery fails.
         */
        [[nodiscard]] bool is_ready();

        /**
         * @brief Wait for the event to be completed.
         *
         * @throws rapidsmpf::cuda_error if cudaEventSynchronize fails.
         */
        void wait();

        /**
         * @brief Get the CUDA event.
         *
         * @return The CUDA event.
         */
        [[nodiscard]] constexpr cudaEvent_t event() const {
            return event_;
        }

      private:
        cudaEvent_t event_;  ///< CUDA event used to track device memory allocation
        std::atomic<bool> done_{false
        };  ///< Cache of the event status to avoid unnecessary queries.
    };

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
    [[nodiscard]] void* data();

    /**
     * @brief Access the underlying memory buffer (host or device memory).
     *
     * @return A const pointer to the underlying host or device memory.
     *
     * @throws std::logic_error if the buffer does not manage any memory.
     */
    [[nodiscard]] void const* data() const;

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
    inline void override_event(std::shared_ptr<Event> event) {
        event_ = std::move(event);
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

    /**
     * @brief Copy data from this buffer to a destination buffer with a given offset.
     *
     * @param dest Destination buffer.
     * @param dest_offset Non-negative offset of the destination buffer (in bytes).
     * @param stream CUDA stream to use for the copy.
     * @param attach_event If true, attach the event to the copy. Else, the caller needs
     * to attach appropriate event to the destination buffer. If the copy is host-to-host,
     * the copy is synchronous and the event is not needed, hence this argument is
     * ignored.
     * @returns Number of bytes written to the destination buffer.
     *
     * @note If this buffer and destination buffer are both on the host, the copy is
     * synchronous.
     *
     * @throws std::invalid_argument if copy violates the bounds of the destination
     * buffer.
     */
    [[nodiscard]] std::ptrdiff_t copy_to(
        Buffer& dest,
        std::ptrdiff_t dest_offset,
        rmm::cuda_stream_view stream,
        bool attach_event = false
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
     * @param br Buffer resource for memory allocation.
     *
     * @throws std::invalid_argument if `host_buffer` is null.
     */
    Buffer(std::unique_ptr<std::vector<uint8_t>> host_buffer, BufferResource* br);

    /**
     * @brief Construct a Buffer from device memory.
     *
     * @param device_buffer A unique pointer to a device buffer.
     * @param stream CUDA stream used for the device buffer allocation.
     * @param br Buffer resource for memory allocation.
     * @param event The shared event to use for the buffer.
     *
     * @throws std::invalid_argument if `device_buffer` is null.
     * @throws std::invalid_argument if `stream` or `br->mr` isn't the same used by
     * `device_buffer`.
     */
    Buffer(
        std::unique_ptr<rmm::device_buffer> device_buffer,
        rmm::cuda_stream_view stream,
        BufferResource* br,
        std::shared_ptr<Event> event = nullptr
    );

    /**
     * @brief Access the underlying host memory buffer.
     *
     * @return A reference to the unique pointer managing the host memory.
     *
     * @throws std::logic_error if the buffer does not manage host memory.
     */
    [[nodiscard]] HostStorageT& host() {
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
    [[nodiscard]] DeviceStorageT& device() {
        if (auto ref = std::get_if<DeviceStorageT>(&storage_)) {
            return *ref;
        } else {
            RAPIDSMPF_FAIL("Buffer is not host memory");
        }
    }

    /**
     * @brief Create a copy of this buffer using the same memory type.
     *
     * @param stream CUDA stream used for the device buffer allocation and copy.
     * @return A unique pointer to a new Buffer containing the copied data.
     */
    [[nodiscard]] std::unique_ptr<Buffer> copy(rmm::cuda_stream_view stream) const;

    /**
     * @brief Create a copy of this buffer using the specified memory type.
     *
     * @param target The target memory type.
     * @param stream CUDA stream used for device buffer allocation and copy.
     * @return A unique pointer to a new Buffer containing the copied data.
     */
    [[nodiscard]] std::unique_ptr<Buffer> copy(
        MemoryType target, rmm::cuda_stream_view stream
    ) const;

  public:
    BufferResource* const br;  ///< The buffer resource used.
    std::size_t const size;  ///< The size of the buffer in bytes.

  private:
    /// @brief The underlying storage host memory or device memory buffer (where
    /// applicable).
    StorageT storage_;
    /// @brief CUDA event used to track copy operations
    std::shared_ptr<Event> event_;
};

}  // namespace rapidsmpf
