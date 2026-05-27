/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <ranges>
#include <unordered_map>
#include <utility>

#include <cuda/memory_resource>

#include <rmm/cuda_stream_pool.hpp>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/detail/buffer_resource_impl.hpp>
#include <rapidsmpf/memory/memory_reservation.hpp>
#include <rapidsmpf/memory/pinned_memory_resource.hpp>
#include <rapidsmpf/memory/resource_types.hpp>
#include <rapidsmpf/memory/scoped_memory_record.hpp>
#include <rapidsmpf/memory/spill_manager.hpp>
#include <rapidsmpf/statistics.hpp>
#include <rapidsmpf/utils/misc.hpp>

namespace rapidsmpf {

/**
 * @brief CCCL-compatible memory resource managing all memory operations in RapidsMPF.
 *
 * `BufferResource` handles allocations and transfers between different memory
 * types (device, host, pinned host). All memory operations in RapidsMPF — for
 * example those performed by the Shuffler — flow through a `BufferResource`.
 *
 * `BufferResource` itself satisfies the CCCL
 * `cuda::mr::resource_with<device_accessible>` concept, so an instance can be
 * passed directly anywhere an RMM-compatible device memory resource is
 * expected (e.g. as the `mr` argument to `rmm::device_buffer`). Buffers
 * allocated through it hold an owning ref to the resource, which transitively
 * keeps the underlying stream pool alive.
 *
 * The class is held by reference-counted shared ownership through
 * `cuda::mr::shared_resource`; copies of a `BufferResource` are cheap and
 * refer to the same underlying state.
 *
 * Memory availability is computed per `MemoryType` as `limit - allocated`.
 * Device and pinned-host allocations routed through this `BufferResource` are
 * tracked automatically. Host memory allocations are not tracked, so the
 * available memory always equals the configured limit. If pinned-host memory
 * is disabled, available pinned-host memory is always reported as zero
 * regardless of the configured limit.
 */
class BufferResource : public cuda::mr::shared_resource<detail::BufferResourceImpl> {
    using shared_base = cuda::mr::shared_resource<detail::BufferResourceImpl>;
    using any_device_resource = cuda::mr::any_resource<cuda::mr::device_accessible>;

  public:
    /// @brief Tag this resource as device-accessible for the CCCL concept.
    friend void get_property(
        BufferResource const&, cuda::mr::device_accessible
    ) noexcept {}

    /**
     * @brief Constructs a buffer resource.
     *
     * @param device_mr The RMM device memory resource used for device allocations.
     * @param pinned_mr The pinned host memory resource used for `MemoryType::PINNED_HOST`
     * allocations. If disabled, pinned host allocations are unavailable regardless of
     * @p memory_limits.
     * @param memory_limits Maximum bytes per memory type before overbooking or spilling.
     * Missing entries default to unlimited.
     * @param periodic_spill_check Enable periodic spill checks. A dedicated thread
     * continuously checks and performs spilling based on memory availability. The value
     * of `periodic_spill_check` controls the pause between checks. If `std::nullopt`, no
     * periodic spill checking is performed.
     * @param stream_pool Pool of CUDA streams used throughout RapidsMPF for operations
     * that do not take an explicit CUDA stream.
     * @param statistics The statistics instance to use (disabled by default).
     */
    BufferResource(
        any_device_resource device_mr,
        std::optional<PinnedMemoryResource> pinned_mr = PinnedMemoryResource::Disabled,
        std::unordered_map<MemoryType, std::int64_t> memory_limits = {},
        std::optional<Duration> periodic_spill_check = std::chrono::milliseconds{1},
        std::shared_ptr<rmm::cuda_stream_pool> stream_pool = std::make_shared<
            rmm::cuda_stream_pool>(16, rmm::cuda_stream::flags::non_blocking),
        std::shared_ptr<Statistics> statistics = Statistics::disabled()
    );

    /**
     * @brief Construct a BufferResource from configuration options.
     *
     * This factory method creates a BufferResource using configuration options to
     * initialize all components.
     *
     * @param mr A device-accessible RMM memory resource.
     * @param options Configuration options.
     * @param statistics The statistics instance to use (disabled by default).
     *
     * @return A shared pointer to a BufferResource instance configured according to the
     * options.
     */
    static std::shared_ptr<BufferResource> from_options(
        any_device_resource mr,
        config::Options options,
        std::shared_ptr<Statistics> statistics = Statistics::disabled()
    );

    /// @brief Default destructor.
    ~BufferResource() noexcept = default;

    /// @brief Default copy constructor (refcounted shared ownership).
    BufferResource(BufferResource const&) noexcept = default;
    /// @brief Default move constructor (refcounted shared ownership).
    BufferResource(BufferResource&&) noexcept = default;
    /**
     * @brief Default copy assignment (refcounted shared ownership).
     * @return Reference to this.
     */
    BufferResource& operator=(BufferResource const&) noexcept = default;
    /**
     * @brief Default move assignment (refcounted shared ownership).
     * @return Reference to this.
     */
    BufferResource& operator=(BufferResource&&) noexcept = default;

    /**
     * @brief Equality by identity.
     *
     * Two `BufferResource` handles are equal iff they share the same underlying
     * impl instance.
     *
     * @param other The other `BufferResource` handle.
     * @return True iff both handles refer to the same impl.
     */
    [[nodiscard]] bool operator==(BufferResource const& other) const noexcept {
        return std::addressof(get()) == std::addressof(other.get());
    }

    // --- Per-allocation tracking (was RmmResourceAdaptor) --------------------

    /**
     * @brief Returns a copy of the main memory record.
     *
     * Lifetime-of-resource allocation statistics, covering all device
     * allocations made through this `BufferResource` since its construction.
     *
     * @return A copy of the main `ScopedMemoryRecord`.
     */
    [[nodiscard]] ScopedMemoryRecord get_main_record() const {
        return get().get_main_record();
    }

    /**
     * @brief Total number of device bytes currently allocated through this resource.
     *
     * @return Currently outstanding allocated bytes.
     */
    [[nodiscard]] std::int64_t current_allocated() const noexcept {
        return get().current_allocated();
    }

    /**
     * @brief Begin a new scoped memory record on the current thread.
     *
     * Pushes a fresh `ScopedMemoryRecord` onto the per-thread record stack.
     * Subsequent allocations and deallocations on this thread are accumulated
     * into the new record (in addition to the main record) until a matching
     * `end_scoped_memory_record()` pops it.
     *
     * @see end_scoped_memory_record()
     */
    void begin_scoped_memory_record() {
        get().begin_scoped_memory_record();
    }

    /**
     * @brief End the topmost scoped memory record on the current thread.
     *
     * Pops the top of the per-thread record stack and returns it. If another
     * scoped record is still active on this thread, the popped record is added
     * to it as a sub-scope.
     *
     * @return The popped `ScopedMemoryRecord`.
     *
     * @throws std::out_of_range if the current thread's record stack is empty.
     *
     * @see begin_scoped_memory_record()
     */
    ScopedMemoryRecord end_scoped_memory_record() {
        return get().end_scoped_memory_record();
    }

    // --- Memory-resource accessors -------------------------------------------

    /**
     * @brief Get the RMM device memory resource.
     *
     * @return Reference to the RMM resource used for device allocations.
     */
    [[nodiscard]] rmm::device_async_resource_ref device_mr() const noexcept {
        // Wrap *this — BufferResource is itself a CCCL-compatible resource.
        // Allocations through the returned ref flow through
        // `shared_resource::allocate` → impl tracker, so they are counted by
        // `current_allocated()`. The const_cast is safe: allocations are
        // logically non-const operations on the underlying state.
        return rmm::device_async_resource_ref{const_cast<BufferResource&>(*this)};
    }

    /**
     * @brief Get the RMM host memory resource.
     *
     * @return Reference to the RMM resource used for host allocations.
     */
    [[nodiscard]] rmm::host_async_resource_ref host_mr() noexcept {
        return get().host_mr();
    }

    /**
     * @brief Get the RMM pinned host memory resource.
     *
     * @throws std::invalid_argument if no pinned memory resource is available.
     * @return Reference to the RMM resource used for pinned host allocations.
     */
    [[nodiscard]] rmm::host_device_async_resource_ref pinned_mr() {
        return get().pinned_mr();
    }

    /**
     * @brief Get the pinned host memory resource if available.
     *
     * @return The pinned host memory resource as an `any_resource`, or
     * `std::nullopt` if pinned host memory is not available.
     */
    [[nodiscard]] std::optional<any_host_device_resource> try_pinned_mr() const noexcept {
        return get().try_pinned_mr();
    }

    // --- Memory availability -------------------------------------------------

    /**
     * @brief Returns the currently available memory for a given memory type, in bytes.
     *
     * Computed as `limit - allocated`. The value may be negative when
     * allocations exceed the configured limit.
     *
     * @param mem_type The memory type to query.
     * @return The available memory in bytes.
     */
    [[nodiscard]] std::int64_t memory_available(MemoryType mem_type) const noexcept {
        return get().memory_available(mem_type);
    }

    /**
     * @brief Updates the memory limit for a given memory type at runtime.
     *
     * The store is atomic, but readers (e.g. `memory_available()` and `reserve()`)
     * observe the limit and the allocation count independently. A concurrent
     * `set_memory_limit()` call can change the limit between a caller's read of
     * `memory_available()` and a subsequent allocation decision; callers that need
     * a coherent view must serialize updates with higher-level synchronization.
     *
     * @param mem_type The memory type whose limit is being updated.
     * @param limit The new byte limit. Negative values are permitted; they make
     * `memory_available(mem_type)` always negative and so trigger continuous
     * spilling.
     */
    void set_memory_limit(MemoryType mem_type, std::int64_t limit) noexcept {
        get().set_memory_limit(mem_type, limit);
    }

    /**
     * @brief Currently reserved bytes for the given memory type.
     *
     * @param mem_type The memory type to query.
     * @return Bytes reserved (but not necessarily allocated) for @p mem_type.
     */
    [[nodiscard]] std::size_t memory_reserved(MemoryType mem_type) const {
        return get().memory_reserved(mem_type);
    }

    /**
     * @brief Reserve an amount of the specified memory type.
     *
     * Creates a new reservation of the specified size and type to inform about upcoming
     * buffer allocations.
     *
     * If overbooking is allowed, a reservation of `size` is returned even when the amount
     * of memory isn't available. In this case, the caller must promise to free buffers
     * corresponding to (at least) the amount of overbooking before using the reservation.
     *
     * If overbooking isn't allowed, a reservation of size zero is returned on failure.
     *
     * @param mem_type The target memory type.
     * @param size The number of bytes to reserve.
     * @param allow_overbooking Whether overbooking is allowed.
     * @return A pair containing the reservation and the amount of overbooking. On success
     * the size of the reservation always equals `size` and on failure the size always
     * equals zero (a zero-sized reservation never fails).
     *
     * @throws std::invalid_argument if the memory type is `MemoryType::PINNED_HOST` and
     * the pinned memory resource is not available.
     */
    std::pair<MemoryReservation, std::size_t> reserve(
        MemoryType mem_type, std::size_t size, AllowOverbooking allow_overbooking
    ) {
        return get().reserve(this, mem_type, size, allow_overbooking);
    }

    /**
     * @brief Reserve device memory and spill if necessary.
     *
     * Attempts to reserve the requested amount of device memory. If insufficient memory
     * is available, spilling is triggered to free up space. When overbooking is allowed,
     * the reservation may succeed even if spilling was not sufficient to fully satisfy
     * the request.
     *
     * @param size The size of the memory to reserve.
     * @param allow_overbooking Whether to allow overbooking. If false, ensures enough
     * memory is freed to satisfy the reservation; otherwise, allows overbooking even
     * if spilling was insufficient.
     * @return The memory reservation.
     *
     * @throws rapidsmpf::reservation_error if allow_overbooking is false and the buffer
     * resource cannot reserve and spill enough device memory.
     */
    MemoryReservation reserve_device_memory_and_spill(
        std::size_t size, AllowOverbooking allow_overbooking
    ) {
        return get().reserve_device_memory_and_spill(this, size, allow_overbooking);
    }

    /**
     * @brief Make a memory reservation or fail based on the given order of memory types.
     *
     * The function attempts to reserve memory by iterating over @p mem_types in the given
     * order of preference. For each memory type, it requests a reservation without
     * overbooking. If no memory type can satisfy the request, the function throws.
     *
     * @param size The size of the buffer to allocate.
     * @param mem_types Range of memory types to try to reserve memory from.
     * @return A memory reservation.
     *
     * @throws std::runtime_error if no memory reservation was made.
     */
    template <std::ranges::input_range Range>
        requires std::convertible_to<std::ranges::range_value_t<Range>, MemoryType>
    [[nodiscard]] MemoryReservation reserve_or_fail(std::size_t size, Range mem_types) {
        for (auto const& mem_type : mem_types) {
            if (mem_type == MemoryType::PINNED_HOST && !get().try_pinned_mr().has_value())
            {
                // Pinned host memory is only available if the memory resource is
                // available.
                continue;
            }
            auto [res, _] = reserve(mem_type, size, AllowOverbooking::NO);
            if (res.size() == size) {
                return std::move(res);
            }
        }
        RAPIDSMPF_FAIL("failed to reserve memory", std::runtime_error);
    }

    /**
     * @brief Make a memory reservation or fail.
     *
     * @param size The size of the buffer to allocate.
     * @param mem_type The single memory type to attempt.
     * @return A memory reservation in @p mem_type.
     *
     * @throws std::runtime_error if no memory reservation was made.
     */
    [[nodiscard]] MemoryReservation reserve_or_fail(
        std::size_t size, MemoryType mem_type
    ) {
        return reserve_or_fail(size, std::ranges::single_view{mem_type});
    }

    /**
     * @brief Consume a portion of the reserved memory.
     *
     * Reduces the remaining size of the reserved memory by the specified amount.
     *
     * @param reservation The reservation to release.
     * @param size The size to consume in bytes.
     * @return The remaining size of the reserved memory after consumption.
     *
     * @throws rapidsmpf::reservation_error if the released size exceeds the size of the
     * reservation.
     */
    std::size_t release(MemoryReservation& reservation, std::size_t size) {
        return get().release(reservation, size);
    }

    // --- Buffer allocation / movement ----------------------------------------

    /**
     * @brief Allocate a buffer of the specified memory type by the reservation.
     *
     * @param size The size of the buffer in bytes.
     * @param stream CUDA stream to use for device allocations.
     * @param reservation The reservation to use for memory allocations.
     * @return A unique pointer to the allocated Buffer.
     *
     * @throws std::invalid_argument if the memory type does not match the reservation.
     * @throws rapidsmpf::reservation_error if `size` exceeds the size of the reservation.
     */
    std::unique_ptr<Buffer> make_buffer(
        std::size_t size, rmm::cuda_stream_view stream, MemoryReservation& reservation
    ) {
        return get().make_buffer(this, size, stream, reservation);
    }

    /**
     * @brief Allocate a buffer consuming the entire reservation.
     *
     * This overload allocates a buffer that matches the full size and memory type
     * of the provided reservation. The reservation is consumed by the call.
     *
     * @param stream CUDA stream to use for device allocations.
     * @param reservation The memory reservation to consume for the allocation.
     * @return A unique pointer to the allocated Buffer.
     */
    std::unique_ptr<Buffer> make_buffer(
        rmm::cuda_stream_view stream, MemoryReservation&& reservation
    ) {
        return get().make_buffer(this, stream, std::move(reservation));
    }

    /**
     * @brief Move device or pinned host buffer data into a Buffer.
     *
     * This operation is cheap; no copy is performed.
     *
     * The resulting Buffer's memory type is inferred from @p data's memory
     * resource: if the resource is host-accessible (e.g. pinned host memory),
     * the Buffer is created with `MemoryType::PINNED_HOST`; otherwise it is
     * created with `MemoryType::DEVICE`.
     *
     * If @p stream differs from the device buffer's current stream:
     *   - @p stream is synchronized with the device buffer's current stream, and
     *   - the device buffer's current stream is updated to @p stream.
     *
     * @param data Unique pointer to the device or pinned host buffer.
     * @param stream CUDA stream associated with the new Buffer. Use or
     * synchronize with this stream when operating on the Buffer.
     * @return Unique pointer to the resulting Buffer.
     */
    std::unique_ptr<Buffer> move(
        std::unique_ptr<rmm::device_buffer> data, rmm::cuda_stream_view stream
    ) {
        return get().move(std::move(data), stream);
    }

    /**
     * @brief Move a Buffer to the memory type specified by the reservation.
     *
     * If the Buffer already resides in the target memory type, a cheap move
     * is performed. Otherwise, the Buffer is copied to the target memory using
     * its own CUDA stream.
     *
     * @param buffer Buffer to move.
     * @param reservation Memory reservation used if a copy is required.
     * @return Unique pointer to the resulting Buffer.
     *
     * @throws rapidsmpf::reservation_error If the allocation size exceeds the
     * reservation.
     */
    std::unique_ptr<Buffer> move(
        std::unique_ptr<Buffer> buffer, MemoryReservation& reservation
    ) {
        return get().move(this, std::move(buffer), reservation);
    }

    /**
     * @brief Move a Buffer to a device buffer.
     *
     * If the Buffer already resides in device memory, a cheap move is performed.
     * Otherwise, the Buffer is copied to device memory using its own CUDA stream.
     *
     * @param buffer The buffer to move.
     * @param reservation Memory reservation used if a copy is required.
     * @return A unique pointer to the resulting device buffer.
     *
     * @throws std::invalid_argument If the reservation's memory type isn't
     * device memory.
     * @throws rapidsmpf::reservation_error if the memory requirement exceeds
     * the reservation.
     */
    std::unique_ptr<rmm::device_buffer> move_to_device_buffer(
        std::unique_ptr<Buffer> buffer, MemoryReservation& reservation
    ) {
        return get().move_to_device_buffer(this, std::move(buffer), reservation);
    }

    /**
     * @brief Move a Buffer into a host buffer.
     *
     * If the Buffer already resides in host memory, a cheap move is performed.
     * Otherwise, the Buffer is copied to host memory using its own CUDA stream.
     *
     * @param buffer Buffer to move.
     * @param reservation Memory reservation used if a copy is required.
     * @return Unique pointer to the resulting host buffer.
     *
     * @throws std::invalid_argument If the reservation's memory type isn't
     * host memory.
     * @throws rapidsmpf::reservation_error If the allocation size exceeds the
     * reservation.
     */
    std::unique_ptr<HostBuffer> move_to_host_buffer(
        std::unique_ptr<Buffer> buffer, MemoryReservation& reservation
    ) {
        return get().move_to_host_buffer(this, std::move(buffer), reservation);
    }

    // --- Stream pool / spill manager / statistics ----------------------------

    /**
     * @brief Returns the CUDA stream pool used by this buffer resource.
     *
     * Use this pool for operations that do not take an explicit CUDA stream.
     *
     * @return Reference to the underlying CUDA stream pool.
     */
    [[nodiscard]] rmm::cuda_stream_pool const& stream_pool() const {
        return get().stream_pool();
    }

    /**
     * @brief Gets a reference to the spill manager used.
     *
     * @return Reference to the SpillManager instance.
     */
    SpillManager& spill_manager() {
        return get().spill_manager();
    }

    /**
     * @brief Gets a shared pointer to the statistics associated with this buffer
     * resource.
     *
     * @return Shared pointer the Statistics instance.
     */
    [[nodiscard]] std::shared_ptr<Statistics> statistics() const noexcept {
        return get().statistics();
    }
};

static_assert(cuda::mr::resource_with<BufferResource, cuda::mr::device_accessible>);
static_assert(StatisticsProvider<BufferResource>);

/**
 * @brief Parse the `spill_device_limit` parameter from configuration options.
 *
 * Reads the `spill_device_limit` option, falling back to 80% of total device
 * memory when unset. The result is aligned down to
 * `rmm::CUDA_ALLOCATION_ALIGNMENT`.
 *
 * @param options Configuration options.
 *
 * @return The device memory limit in bytes.
 */
std::int64_t device_limit_from_options(config::Options options);

/**
 * @brief Get the `periodic_spill_check` parameter from configuration options.
 *
 * @param options Configuration options.
 *
 * @return The duration of the pause between spill checks or std::nullopt if no dedicated
 * thread should check for spilling.
 */
std::optional<Duration> periodic_spill_check_from_options(config::Options options);

/**
 * @brief Get a new CUDA stream pool from configuration options.
 *
 * @param options Configuration options.
 * @return Pool of CUDA streams used throughout RapidsMPF for operations that do
 * not take an explicit CUDA stream.
 */
std::shared_ptr<rmm::cuda_stream_pool> stream_pool_from_options(config::Options options);


}  // namespace rapidsmpf
