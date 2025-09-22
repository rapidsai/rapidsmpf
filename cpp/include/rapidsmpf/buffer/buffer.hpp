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
     */
    [[nodiscard]] HostStorageT const& host() const;

    /**
     * @brief Access the underlying device memory buffer (const).
     *
     * @return A const reference to the unique pointer managing the device memory.
     *
     * @throws std::logic_error if the buffer does not manage device memory.
     */
    [[nodiscard]] DeviceStorageT const& device() const;

    /**
     * @brief Access the underlying memory buffer (host or device memory).
     *
     * @return A const pointer to the underlying host or device memory.
     *
     * @throws std::logic_error if the buffer does not manage any memory.
     */
    [[nodiscard]] std::byte const* data() const;

    /**
     * @brief Provides write access to the buffer.
     *
     * Calls @p f with a pointer to the buffer's memory. The callable must be invocable
     * as `R(std::byte*)`; its return value (if any) is returned by this function.
     *
     * The provided @p stream is the stream associated with this access. Any work enqueued
     * on the buffer memory must use @p stream or synchronize with it before @p f returns.
     * Synchronizing with @p stream only after @p f returns is not sufficient and results
     * in undefined behavior. Normally, @p stream should be the buffer's own stream.
     *
     * @warning The pointer is valid only for the duration of the call. Using it outside
     * of @p f is undefined behavior.
     *
     * @tparam F Callable type.
     * @param stream CUDA stream to use or synchronize with during buffer access.
     * @param f Callable that accepts a single `std::byte*`.
     * @return Whatever @p f returns (`void` if none).
     *
     * @code{.cpp}
     * // Snippet: copy data from `src_ptr` into `buffer`.
     * buffer.write_access(stream, [&](std::byte* buffer_ptr) {
     *     RAPIDSMPF_CUDA_TRY_ALLOC(cudaMemcpyAsync(
     *         buffer_ptr,
     *         src_ptr,
     *         num_bytes,
     *         cudaMemcpyDefault,
     *         stream
     *     ));
     * });
     * @endcode
     */
    template <typename F>
    auto write_access(rmm::cuda_stream_view stream, F&& f)
        -> std::invoke_result_t<F, std::byte*> {
        static_assert(
            std::is_invocable_v<std::remove_reference_t<F>, std::byte*>,
            "write_access() expects a callable with signature: R(std::byte*)"
        );
        using R = std::invoke_result_t<F, std::byte*>;

        // After `f()` completes, an event is recorded on `stream`; this becomes the new
        // latest-write event.
        if constexpr (std::is_void_v<R>) {
            std::invoke(std::forward<F>(f), const_cast<std::byte*>(data()));
            latest_write_event_.record(stream);
        } else {
            auto ret = std::invoke(std::forward<F>(f), const_cast<std::byte*>(data()));
            latest_write_event_.record(stream);
            return ret;
        }
    }

    /**
     * @brief Acquire non-stream-ordered exclusive access to the buffer's memory.
     *
     * Alternative to `write_access()`. Acquires an internal exclusive lock so that
     * **any other access through the Buffer API** (including `write_access()`) will
     * fail with `std::logic_error` while the lock is held. The lock remains held
     * until `unlock()` is called.
     *
     * Use this when integrating with non-stream-aware consumer APIs that require a
     * raw pointer and cannot be expressed as work on a CUDA stream (e.g., MPI,
     * blocking host I/O).
     *
     * @note Prefer `write_access(stream, ...)` if you can express the operation as a
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
     * @brief Check whether the buffer is currently exclusively locked.
     *
     * @return `true` if `exclusive_data_access()` has acquired the lock and `unlock()`
     * has not yet been called; `false` otherwise.
     */
    [[nodiscard]] bool is_locked() const {
        return lock_.load(std::memory_order_acquire);
    }

    /**
     * @brief Release the exclusive lock acquired by `exclusive_data_access()`.
     *
     * @post `is_locked() == false`.
     */
    void unlock() {
        lock_.store(false, std::memory_order_release);
    }

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
     *   cudaStreamSynchronize(buffer.stream());
     *   MPI_Isend(buffer.data(), buffer.size(), MPI_BYTE, dst, tag, comm, &req);
     * }
     * @endcode
     */
    [[nodiscard]] bool is_latest_write_done() const {
        RAPIDSMPF_EXPECTS(!is_locked(), "the buffer is locked");
        return latest_write_event_.is_ready();
    }

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
    [[nodiscard]] DeviceStorageT release_device() {
        RAPIDSMPF_EXPECTS(!is_locked(), "the buffer is locked");
        return std::move(device());
    }

    /**
     * @brief Release the underlying host memory buffer.
     *
     * @return The underlying host memory buffer.
     *
     * @throws std::logic_error if the buffer does not manage host memory.
     * @throws std::logic_error If the buffer is locked.
     */
    [[nodiscard]] HostStorageT release_host() {
        RAPIDSMPF_EXPECTS(!is_locked(), "the buffer is locked");
        return std::move(host());
    }

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
    Buffer& src,
    std::size_t size,
    std::ptrdiff_t dst_offset = 0,
    std::ptrdiff_t src_offset = 0
);

}  // namespace rapidsmpf
