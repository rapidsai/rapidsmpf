/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <variant>

#include <cuda_runtime.h>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <rapidsmpf/cuda_event.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/fixed_sized_host_buffer.hpp>
#include <rapidsmpf/memory/host_buffer.hpp>
#include <rapidsmpf/memory/memory_type.hpp>
#include <rapidsmpf/statistics.hpp>
#include <rapidsmpf/utils/misc.hpp>

namespace rapidsmpf {

/**
 * @brief Buffer representing device or host memory.
 *
 * A `Buffer` holds either device memory or host memory, determined by its memory type
 * at construction. See `device_buffer_types` and `host_buffer_types` for the sets of
 * memory types that result in device-backed and host-backed storage.
 *
 * Buffers are stream ordered and have an associated CUDA stream (see `stream()`). All
 * work that reads or writes the buffer must either be enqueued on that stream or be
 * synchronized with it before accessing the memory. For example, when passing the buffer
 * to a non-stream aware API (e.g., MPI or host-only code), the caller must ensure that
 * the most recent write has completed before the hand off. This can be done by
 * synchronizing the buffer's stream or by checking `is_latest_write_done()`.
 *
 * To obtain an `rmm::device_buffer` from a `Buffer`, first ensure that the buffer's
 * memory type is one of the types listed in `device_buffer_types` (moving the buffer if
 * necessary), then call `release_device_buffer()`.
 *
 * @note The constructors are private. Buffers are created through `BufferResource`.
 */
class Buffer {
    friend class BufferResource;

  public:
    /// @brief Storage type for a device buffer.
    using DeviceBufferT = std::unique_ptr<rmm::device_buffer>;

    /// @brief Storage type for a host buffer.
    using HostBufferT = std::unique_ptr<HostBuffer>;

    /// @brief Storage type for a pinned host buffer backed by fixed-size blocks.
    using FixedSizedHostBufferT = std::unique_ptr<FixedSizedHostBuffer>;

    /**
     * @brief Memory types suitable for constructing a device backed buffer.
     *
     * A buffer may use `DeviceBufferT` only if its memory type is listed here.
     * This ensures that the buffer is backed by memory that behaves as device
     * accessible memory.
     */
    static constexpr std::array<MemoryType, 1> device_buffer_types{MemoryType::DEVICE};

    /**
     * @brief Memory types suitable for constructing a host backed buffer.
     *
     * A buffer may use `HostBufferT` only if its memory type is listed here.
     * This ensures that the buffer is backed by memory that behaves as host
     * accessible memory.
     */
    static constexpr std::array<MemoryType, 2> host_buffer_types{
        MemoryType::HOST, MemoryType::PINNED_HOST
    };

    /**
     * @brief Memory types suitable for constructing a pinned host buffer backed
     * by fixed-size blocks.
     *
     * A buffer may use `FixedSizedHostBufferT` only if its memory type is listed here.
     */
    static constexpr std::array<MemoryType, 1> pinned_buffer_types{
        MemoryType::PINNED_HOST
    };

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
     * @brief Provides stream-ordered write access to the buffer's memory as a
     * sequence of contiguous blocks.
     *
     * Like `write_access()`, this is a stream-ordered operation: all work
     * performed by @p f must be ordered on the buffer's stream. After all
     * blocks have been visited, a write event is recorded on the stream.
     *
     * Unlike `write_access()`, this method works for **all** storage types:
     *
     * - **DEVICE / HOST** (contiguous): @p f is called once with a span
     *   covering the entire allocation.
     * - **PINNED_HOST** (`FixedSizedHostBuffer`): @p f is called once per
     *   fixed-size block, in order.
     *
     * The callable must be invocable as:
     *   - `void(std::span<std::byte> block, rmm::cuda_stream_view stream)`.
     *
     * @warning Each span is valid only for the duration of its individual call.
     *
     * @tparam F Callable type.
     * @param f Callable that accepts `(std::span<std::byte>, rmm::cuda_stream_view)`.
     *
     * @throws std::logic_error If the buffer is locked.
     *
     * @see write_access()
     */
    template <typename F>
    void write_access_blocks(F&& f) {
        using Fn = std::remove_reference_t<F>;
        static_assert(
            std::is_invocable_v<Fn, std::span<std::byte>, rmm::cuda_stream_view>,
            "write_access_blocks() expects callable void(std::span<std::byte>, "
            "rmm::cuda_stream_view)"
        );

        throw_if_locked();

        std::visit(
            overloaded{
                [&](FixedSizedHostBufferT& buf) {
                    for (auto block : buf->blocks()) {
                        std::invoke(
                            f, std::span<std::byte>{block, buf->block_size()}, stream_
                        );
                    }
                },
                [&](auto& buf) {
                    std::invoke(
                        std::forward<F>(f),
                        std::span<std::byte>{
                            reinterpret_cast<std::byte*>(buf->data()), buf->size()
                        },
                        stream_
                    );
                },
            },
            storage_
        );

        latest_write_event_.record(stream_);
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
     * @warning The `Buffer` does not track read access to its underlying storage, and so
     * one should be aware of write-after-read anti-dependencies when obtaining exclusive
     * access.
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
     * @brief Acquire non-stream-ordered exclusive access to the buffer's memory
     * as a list of block-start pointers.
     *
     * Like `exclusive_data_access()`, acquires the internal exclusive lock until
     * `unlock()` is called. Unlike `exclusive_data_access()`, this method works
     * for **all** storage types:
     *
     * - **DEVICE / HOST** (contiguous): returns a single-element vector whose
     *   one pointer is the start of the contiguous allocation.
     * - **PINNED_HOST** (`FixedSizedHostBuffer`): returns one pointer per
     *   fixed-size block (equivalent to `FixedSizedHostBuffer::blocks()`).
     *
     * The pointers remain valid until `unlock()` is called.
     *
     * @return Vector of block-start pointers.
     *
     * @throws std::logic_error If the buffer is already locked.
     * @throws std::logic_error If `is_latest_write_done() != true`.
     *
     * @see exclusive_data_access(), write_access_blocks(), unlock()
     */
    std::vector<std::byte*> exclusive_data_access_blocks();

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
        return mem_type_;
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
     * @brief Get the CUDA event that tracks the latest write into the buffer.
     *
     * @return The CUDA event that tracks the latest write into the buffer.
     */
    [[nodiscard]] CudaEvent const& latest_write_event() const noexcept {
        return latest_write_event_;
    }

    /**
     * @brief Rebind the buffer to a new CUDA stream.
     *
     * Changes the buffer's associated stream to @p new_stream and ensures proper
     * synchronization: @p new_stream will wait for any pending work on the current
     * stream before proceeding. The underlying storage stream (e.g., the stream of
     * an `rmm::device_buffer` or `HostBuffer`) is also updated.
     *
     * @param new_stream The new CUDA stream.
     *
     * @throws std::logic_error If the buffer is locked.
     *
     * @code{.cpp}
     * // Example: merge buffers from different streams onto a single stream.
     * Buffer buffer_a = ...;  // associated with stream_a
     * Buffer buffer_b = ...;  // associated with stream_b
     *
     * buffer_a.rebind_stream(merged_stream);
     * buffer_b.rebind_stream(merged_stream);
     *
     * // Both buffers now use merged_stream with proper synchronization
     * buffer_copy(buffer_a, buffer_b, size);
     * @endcode
     */
    void rebind_stream(rmm::cuda_stream_view new_stream);

    /**
     * @brief Asynchronously copy data from this buffer into @p dst.
     *
     * Copies @p size bytes from this buffer at @p src_offset into @p dst at @p
     * dst_offset.
     *
     * @param dst Destination buffer (must not be `*this`).
     * @param size Number of bytes to copy.
     * @param dst_offset Offset (in bytes) into the destination buffer.
     * @param src_offset Offset (in bytes) into this (source) buffer.
     * @param statistics Statistics object used to record the copy operation. Pass
     * `nullptr` or `Statistics::disabled()` to skip recording.
     *
     * @throws std::invalid_argument If @p dst is the same object as `*this`.
     * @throws std::invalid_argument If the copy range is out of bounds for either buffer.
     */
    void copy_to(
        Buffer& dst,
        std::size_t size,
        std::ptrdiff_t dst_offset = 0,
        std::ptrdiff_t src_offset = 0,
        std::shared_ptr<Statistics> statistics = std::make_shared<Statistics>(false)
    ) const;

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
     * @warning This check only confirms that there are no pending _writes_ to the
     * `Buffer`. Pending stream-ordered _reads_ from the `Buffer` are not tracked and
     * therefore one should be aware of write-after-read anti-dependencies when using this
     * check to pass from stream-ordered to non-stream-ordered code.
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
     * @brief Construct a stream-ordered Buffer from synchronized host buffer.
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
     * @param mem_type The memory type of the underlying @p host_buffer.
     *
     * @throws std::invalid_argument If @p host_buffer is null.
     * @throws std::logic_error If the buffer is locked, or @p mem_type is not suitable
     * for @p host_buffer (see warning for details).
     *
     * @warning The caller is responsible to ensure @p mem_type is suitable for @p
     * host_buffer. An unsuitable memory type leads to an irrecoverable condition.
     */
    Buffer(
        std::unique_ptr<HostBuffer> host_buffer,
        rmm::cuda_stream_view stream,
        MemoryType mem_type
    );

    /**
     * @brief Construct a stream-ordered Buffer from a device buffer.
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
     * @param mem_type The memory type of the underlying @p device_buffer.
     *
     * @throws std::invalid_argument If @p device_buffer is null.
     * @throws std::logic_error If the buffer is locked, or @p mem_type is not suitable
     * for @p device_buffer (see warning for details).
     *
     * @warning The caller is responsible to ensure @p mem_type is suitable for @p
     * device_buffer. An unsuitable memory type leads to an irrecoverable condition.
     */
    Buffer(std::unique_ptr<rmm::device_buffer> device_buffer, MemoryType mem_type);

    /**
     * @brief Construct a stream-ordered Buffer from a fixed-sized host buffer.
     *
     * Adopts @p fixed_host_buffer as the Buffer's storage and associates the Buffer
     * with @p stream for subsequent stream-ordered operations.
     *
     * @note The constructor does **not** perform any synchronization. The caller must
     * ensure that @p fixed_host_buffer is already synchronized at the time of
     * construction.
     *
     * @warning Many `Buffer` APIs (e.g., `data()`, `exclusive_data_access()`,
     * `rebind_stream()`) are **not supported** for `FixedSizedHostBuffer`-backed
     * buffers and will throw `std::logic_error`.
     *
     * @param fixed_host_buffer Unique pointer to a FixedSizedHostBuffer.
     * @param size The logical size in bytes of the data. This may be smaller than
     *   `fixed_host_buffer->total_size()` because the underlying allocation is
     *   rounded up to a block-size boundary.
     * @param stream CUDA stream to associate with the Buffer.
     * @param mem_type The memory type (must be in `pinned_buffer_types`).
     *
     * @throws std::invalid_argument If @p fixed_host_buffer is null.
     * @throws std::invalid_argument If @p size exceeds `fixed_host_buffer->total_size()`.
     * @throws std::logic_error If @p mem_type is not suitable for a pinned buffer.
     */
    Buffer(
        std::unique_ptr<FixedSizedHostBuffer> fixed_host_buffer,
        std::size_t size,
        rmm::cuda_stream_view stream,
        MemoryType mem_type
    );

    /**
     * @brief Throws if the buffer is currently locked by `exclusive_data_access()`.
     *
     * @throws std::logic_error If the buffer is locked.
     */
    void throw_if_locked() const;

    /**
     * @brief Release the underlying device buffer.
     *
     * @return The underlying device buffer.
     *
     * @throws std::logic_error if the buffer does not manage a device buffer.
     * @throws std::logic_error If the buffer is locked.
     */
    [[nodiscard]] DeviceBufferT release_device_buffer();

    /**
     * @brief Release the underlying host buffer.
     *
     * @return The underlying host buffer.
     *
     * @throws std::logic_error if the buffer does not manage a host buffer.
     * @throws std::logic_error If the buffer is locked.
     */
    [[nodiscard]] HostBufferT release_host_buffer();

    /**
     * @brief Release the underlying fixed-sized host buffer.
     *
     * @return The underlying fixed-sized host buffer.
     *
     * @throws std::logic_error if the buffer does not manage a FixedSizedHostBuffer.
     * @throws std::logic_error If the buffer is locked.
     */
    [[nodiscard]] FixedSizedHostBufferT release_fixed_sized_host_buffer();

  public:
    std::size_t const size;  ///< The size of the buffer in bytes.

  private:
    MemoryType const mem_type_;
    std::variant<DeviceBufferT, HostBufferT, FixedSizedHostBufferT> storage_;
    rmm::cuda_stream_view stream_;
    CudaEvent latest_write_event_;
    std::atomic<bool> lock_;
};

/**
 * @brief Asynchronously copy data between buffers.
 *
 * Copies @p size bytes from @p src, starting at @p src_offset, into @p dst at
 * @p dst_offset.
 *
 * @param statistics Statistics object used to record the copy operation. Use
 * `Statistics::disabled()` to skip recording.
 * @param dst Destination buffer.
 * @param src Source buffer.
 * @param size Number of bytes to copy.
 * @param dst_offset Byte offset into the destination buffer.
 * @param src_offset Byte offset into the source buffer.
 *
 * @throws std::invalid_argument If the requested range is out of bounds.
 */
void buffer_copy(
    std::shared_ptr<Statistics> statistics,
    Buffer& dst,
    Buffer const& src,
    std::size_t size,
    std::ptrdiff_t dst_offset = 0,
    std::ptrdiff_t src_offset = 0
);

}  // namespace rapidsmpf
