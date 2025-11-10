/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <variant>
#include <vector>

#include <cuda_runtime.h>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <rapidsmpf/cuda_event.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/utils.hpp>

namespace rapidsmpf {

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
 * @note This buffer is stream-ordered and has an associated CUDA stream (see `stream()`).
 * All work (host and device) that reads or writes the buffer must either be enqueued on
 * that stream or be synchronized with it *before* accessing the memory.
 * @note When passing the buffer to a non-stream-aware API (e.g., MPI, host-only code),
 * you must ensure the last write has completed *before* the hand-off. Either synchronize
 * the buffer's stream (e.g., `stream().synchronize()`) or verify completion via
 * `is_latest_write_done()`.
 */
class Buffer {
    friend class BufferResource;

  public:
    /// @brief Storage type for the device buffer.
    using DeviceStorageT = std::unique_ptr<rmm::device_buffer>;

    /// @brief Storage type for the host buffer.
    using HostStorageT = std::unique_ptr<std::vector<uint8_t>>;

    /**
     * @brief Storage type in Buffer, which could be either host or device memory.
     */
    using StorageT = std::variant<DeviceStorageT, HostStorageT>;

    /**
     * @brief Access the underlying host memory buffer (const).
     *
     * @return A reference to the unique pointer managing the host memory.
     *
     * @throws std::logic_error if the buffer does not manage host memory.
     * @throws std::logic_error If the buffer is locked.
     */
    [[nodiscard]] HostStorageT const& host() const;

    /**
     * @brief Access the underlying device memory buffer (const).
     *
     * @return A const reference to the unique pointer managing the device memory.
     *
     * @throws std::logic_error if the buffer does not manage device memory.
     * @throws std::logic_error If the buffer is locked.
     */
    [[nodiscard]] DeviceStorageT const& device() const;

    /**
     * @brief Access the underlying memory buffer (host or device memory).
     *
     * @return A const pointer to the underlying host or device memory.
     *
     * @throws std::logic_error if the buffer does not manage any memory.
     * @throws std::logic_error If the buffer is locked.
     */
    [[nodiscard]] std::byte const* data() const;

    /**
     * @brief Provides stream-ordered write access to the buffer.
     *
     * Calls @p f with a pointer to the buffer's memory and the buffer's stream
     * (i.e., `this->stream()`).
     *
     * The callable must be invocable as:
     *   - `R(std::byte*, rmm::cuda_stream_view)`.
     *
     * All work performed by @p f must be stream-ordered on the buffer's stream.
     * Enqueuing work on any other stream without synchronizing with the buffer's
     * stream before and after the call is undefined behavior. In other words,
     * @p f must behave as a single stream-ordered operation, similar to issuing one
     * `cudaMemcpyAsync` on the buffer's stream. For non-stream-aware integrations,
     * use `exclusive_data_access()`.
     *
     * After @p f returns, an event is recorded on the buffer's stream, establishing
     * the new "latest write" for this buffer.
     *
     * @warning The pointer is valid only for the duration of the call. Using it
     * outside of @p f is undefined behavior.
     *
     * @tparam F Callable type.
     * @param f Callable that accepts `(std::byte*, rmm::cuda_stream_view)`.
     * @return Whatever @p f returns (`void` if none).
     *
     * @throws std::logic_error If the buffer is locked.
     *
     * @code{.cpp}
     * // Snippet: copy data from `src_ptr` into `buffer` on the buffer's stream.
     * buffer.write_access([&](std::byte* buffer_ptr, rmm::cuda_stream_view stream) {
     *   assert(buffer.stream().value() == stream.value());
     *   RAPIDSMPF_CUDA_TRY_ALLOC(cudaMemcpyAsync(
     *       buffer_ptr,
     *       src_ptr,
     *       num_bytes,
     *       cudaMemcpyDefault,
     *       stream
     *   ));
     * });
     * @endcode
     */
    template <typename F>
    auto write_access(F&& f)
        -> std::invoke_result_t<F, std::byte*, rmm::cuda_stream_view> {
        using Fn = std::remove_reference_t<F>;
        static_assert(
            std::is_invocable_v<Fn, std::byte*, rmm::cuda_stream_view>,
            "write_access() expects callable R(std::byte*, rmm::cuda_stream_view)"
        );
        using R = std::invoke_result_t<Fn, std::byte*, rmm::cuda_stream_view>;

        auto* ptr = const_cast<std::byte*>(data());
        if constexpr (std::is_void_v<R>) {
            std::invoke(std::forward<F>(f), ptr, stream_);
            latest_write_event_.record(stream_);
        } else {
            auto ret = std::invoke(std::forward<F>(f), ptr, stream_);
            latest_write_event_.record(stream_);
            return ret;
        }
    }

    /**
     * @brief Acquire non-stream-ordered exclusive access to the buffer's memory.
     *
     * Alternative to `write_access()`. Acquires an internal exclusive lock so that
     * **any other access through the Buffer API** (including `write_access()`) will
     * fail with `std::logic_error` while the lock is held. The lock remains held
     * until `unlock()` is called. This lock is not a concurrency mechanism; it only
     * prevents accidental access to the Buffer through the rest of the Buffer API while
     * locked.
     *
     * Use this when integrating with non-stream-aware consumer APIs that require a
     * raw pointer and cannot be expressed as work on a CUDA stream (e.g., MPI, blocking
     * host I/O).
     *
     * @note Prefer `write_access(...)` if you can express the operation as a
     * single callable on a stream, even if that requires manually synchronizing the
     * stream before the callable returns.
     *
     * @return Pointer to the underlying storage.
     *
     * @throws std::logic_error If the buffer is already locked.
     * @throws std::logic_error If `is_latest_write_done() != true`.
     *
     * @see write_access(), is_locked(), unlock()
     */
    std::byte* exclusive_data_access();

    /**
     * @brief Release the exclusive lock acquired by `exclusive_data_access()`.
     */
    void unlock();

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
                [](HostStorageT const&) -> MemoryType { return MemoryType::HOST; },
                [](DeviceStorageT const&) -> MemoryType { return MemoryType::DEVICE; }
            },
            storage_
        );
    }

    /**
     * @brief Get the associated CUDA stream.
     *
     * All operations must either use this stream or synchronize with it
     * before accessing the underlying data (both host and device memory).
     *
     * @return The associated CUDA stream.
     */
    [[nodiscard]] constexpr rmm::cuda_stream_view stream() const noexcept {
        return stream_;
    }

    /**
     * @brief Check whether the buffer's most recent write has completed.
     *
     * Returns whether the CUDA event that tracks the most recent write into this
     * buffer has been signaled.
     *
     * Use this to guard *non-stream-ordered* consumer-APIs that do not accept a CUDA
     * stream (e.g., MPI sends/receives, host-side reads).
     *
     * @note This is a non-blocking, point-in-time status check and is subject to TOCTOU
     * races: another thread may enqueue additional writes after this returns `true`.
     * Ensure no further writes are enqueued, or establish stronger synchronization (e.g.,
     * synchronize the buffer's stream) before using the buffer.
     *
     * @return `true` if the last recorded write event has completed; `false` otherwise.
     *
     * @throws std::logic_error If the buffer is locked.
     *
     * @code{.cpp}
     * // Example: send the buffer via MPI (non-stream-ordered).
     * if (buffer.is_latest_write_done()) {
     *   MPI_Isend(buffer.data(), buffer.size(), MPI_BYTE, dst, tag, comm, &req);
     * } else {
     *   // Ensure completion before handing to MPI.
     *   buffer.stream().synchronize();
     *   MPI_Isend(buffer.data(), buffer.size(), MPI_BYTE, dst, tag, comm, &req);
     * }
     * @endcode
     */
    [[nodiscard]] bool is_latest_write_done() const;

    /// @brief Delete move and copy constructors and assignment operators.
    Buffer(Buffer&&) = delete;
    Buffer(Buffer const&) = delete;
    Buffer& operator=(Buffer& o) = delete;
    Buffer& operator=(Buffer&& o) = delete;

  private:
    /**
     * @brief Construct a stream-ordered Buffer from synchronized host memory.
     *
     * Adopts @p host_buffer as the Buffer's storage and associates the Buffer with
     * @p stream for subsequent stream-ordered operations.
     *
     * @note The constructor does **not** perform any synchronization. The caller must
     * ensure that @p host_buffer is already synchronized (no pending GPU or stream work)
     * at the time of construction. A newly constructed Buffer is therefore considered
     * ready (i.e., `is_latest_write_done() == true`).
     *
     * @param host_buffer Unique pointer to a vector containing host memory.
     * @param stream CUDA stream to associate with the Buffer for future operations.
     *
     * @throws std::invalid_argument If @p host_buffer is null.
     * @throws std::logic_error If the buffer is locked.
     */
    Buffer(
        std::unique_ptr<std::vector<uint8_t>> host_buffer, rmm::cuda_stream_view stream
    );

    /**
     * @brief Construct a stream-ordered Buffer from device memory.
     *
     * Adopts @p device_buffer as the Buffer's storage and inherits its CUDA stream.
     * At construction, the Buffer records an initial "latest write" on that stream,
     * so `is_latest_write_done()` will become `true` once all work enqueued on the
     * adopted stream up to this point has completed.
     *
     * @note No synchronization is performed by the constructor. Any producer that
     * initialized or modified @p device_buffer must have enqueued that work on the same
     * stream (or established ordering with it) for correctness.
     *
     * @param device_buffer Unique pointer to a device buffer. Must be non-null.
     *
     * @throws std::invalid_argument If @p device_buffer is null.
     * @throws std::logic_error If the buffer is locked.
     */
    Buffer(std::unique_ptr<rmm::device_buffer> device_buffer);

    /**
     * @brief Throws if the buffer is currently locked by `exclusive_data_access()`.
     *
     * @throws std::logic_error If the buffer is locked.
     */
    void throw_if_locked() const;

    /**
     * @brief Access the underlying host memory buffer.
     *
     * @return A reference to the unique pointer managing the host memory.
     *
     * @throws std::logic_error if the buffer does not manage host memory.
     * @throws std::logic_error If the buffer is locked.
     */
    [[nodiscard]] HostStorageT& host();

    /**
     * @brief Access the underlying device memory buffer.
     *
     * @return A reference to the unique pointer managing the device memory.
     *
     * @throws std::logic_error if the buffer does not manage device memory.
     * @throws std::logic_error If the buffer is locked.
     */
    [[nodiscard]] DeviceStorageT& device();

    /**
     * @brief Release the underlying device memory buffer.
     *
     * @return The underlying device memory buffer.
     *
     * @throws std::logic_error if the buffer does not manage device memory.
     * @throws std::logic_error If the buffer is locked.
     */
    [[nodiscard]] DeviceStorageT release_device();

    /**
     * @brief Release the underlying host memory buffer.
     *
     * @return The underlying host memory buffer.
     *
     * @throws std::logic_error if the buffer does not manage host memory.
     * @throws std::logic_error If the buffer is locked.
     */
    [[nodiscard]] HostStorageT release_host();

  public:
    std::size_t const size;  ///< The size of the buffer in bytes.

  private:
    /// @brief The underlying storage host memory or device memory buffer (where
    /// applicable).
    StorageT storage_;
    rmm::cuda_stream_view stream_;
    CudaEvent latest_write_event_;
    std::atomic_bool lock_;
};

/**
 * @brief Asynchronously copy data between buffers.
 *
 * Copies @p size bytes from @p src at @p src_offset into @p dst at @p dst_offset.
 *
 * @param dst Destination buffer.
 * @param src Source buffer.
 * @param size Number of bytes to copy.
 * @param dst_offset Offset (in bytes) into the destination buffer.
 * @param src_offset Offset (in bytes) into the source buffer.
 *
 * @throws std::invalid_argument If out of bounds.
 */
void buffer_copy(
    Buffer& dst,
    Buffer const& src,
    std::size_t size,
    std::ptrdiff_t dst_offset = 0,
    std::ptrdiff_t src_offset = 0
);

}  // namespace rapidsmpf
