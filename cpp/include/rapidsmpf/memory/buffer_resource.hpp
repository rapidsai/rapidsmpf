/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <ranges>
#include <unordered_map>
#include <utility>

#include <cuda/memory_resource>
#include <cuda/stream>

#include <rmm/cuda_stream_pool.hpp>

#include <rapidsmpf/disk/disk_resource.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/host_memory_resource.hpp>
#include <rapidsmpf/memory/memory_reservation.hpp>
#include <rapidsmpf/memory/pinned_memory_resource.hpp>
#include <rapidsmpf/memory/resource_types.hpp>
#include <rapidsmpf/memory/spill_manager.hpp>
#include <rapidsmpf/rmm_resource_adaptor.hpp>
#include <rapidsmpf/statistics.hpp>
#include <rapidsmpf/utils/misc.hpp>

namespace rapidsmpf {

/**
 * @brief Policy controlling whether a memory reservation is allowed to overbook.
 *
 * This enum is used throughout RapidsMPF to specify the overbooking behavior of
 * a memory reservation request. The exact semantics depend on the specific API
 * and execution context in which it is used.
 */
enum class AllowOverbooking : bool {
    NO,  ///< Overbooking is not allowed.
    YES,  ///< Overbooking is allowed.
};

/**
 * @brief Pool of non-blocking CUDA streams.
 *
 * The pool is backed by RMM until CUDA Core provides an owning stream-pool
 * implementation. Its public interface returns CUDA Core stream references so
 * callers do not need to bridge between stream abstractions.
 */
class StreamPool {
  public:
    /**
     * @brief Construct a pool of non-blocking CUDA streams.
     *
     * @param num_streams Number of streams to create.
     */
    explicit StreamPool(std::size_t num_streams)
        : pool_{std::make_shared<rmm::cuda_stream_pool>(
              num_streams, rmm::cuda_stream::flags::non_blocking
          )} {}

    /**
     * @brief Construct a stream pool backed by an existing RMM pool.
     *
     * @param pool RMM stream pool providing stream ownership.
     */
    explicit StreamPool(std::shared_ptr<rmm::cuda_stream_pool> pool)
        : pool_{std::move(pool)} {
        RAPIDSMPF_EXPECTS(pool_ != nullptr, "the stream pool pointer cannot be NULL");
    }

    /**
     * @brief Get the next stream from the pool.
     *
     * @return A CUDA Core reference to the selected stream.
     */
    [[nodiscard]] cuda::stream_ref get_stream() const {
        return pool_->get_stream();
    }

    /**
     * @brief Get a stream from the pool by index.
     *
     * @param stream_id Index of the stream to retrieve.
     * @return A CUDA Core reference to the selected stream.
     */
    [[nodiscard]] cuda::stream_ref get_stream(std::size_t stream_id) const {
        return pool_->get_stream(stream_id);
    }

    /**
     * @brief Get the number of streams in the pool.
     *
     * @return Number of streams in the pool.
     */
    [[nodiscard]] std::size_t get_pool_size() const {
        return pool_->get_pool_size();
    }

  private:
    std::shared_ptr<rmm::cuda_stream_pool> pool_;
};

/**
 * @brief Class managing buffer resources.
 *
 * This class handles memory allocation and transfers between different memory types
 * (e.g., host and device). All memory operations in rapidsmpf, such as those performed
 * by the Shuffler, rely on a buffer resource for memory management.
 *
 * @note `BufferResource` instances must be constructed through `create()` or
 * `from_options()`, both of which return a `std::shared_ptr<BufferResource>`. Direct
 * construction is disabled.
 *
 * @note Allocation tracking only applies to allocations routed through this
 * `BufferResource`. The constructor wraps the supplied device memory resource
 * in an internal adaptor that records all allocations and deallocations; that
 * adaptor is exposed via `device_mr()`.
 *
 * Allocations made through the original, unwrapped memory resource bypass
 * this tracking and are therefore invisible to memory-limit accounting and
 * statistics.
 *
 * To ensure all CUDA allocations count against the `BufferResource` budget,
 * use `br->device_mr()` everywhere instead of the underlying memory resource
 * passed to the constructor.
 *
 * Tracking allocations made outside `BufferResource`, for example allocations
 * performed before construction or through code paths that use a raw memory
 * resource directly, is a separate design concern and is not handled by this
 * class.
 */
class BufferResource : public std::enable_shared_from_this<BufferResource> {
  public:
    /**
     * @brief Construct a `BufferResource` managed by `std::shared_ptr`.
     *
     * Available memory is computed per `MemoryType` as `limit - allocated`.
     *
     * Device and pinned-host allocations routed through this `BufferResource` are tracked
     * automatically. Host memory allocations are not tracked and therefore always report
     * the configured limit as available memory.
     *
     * If pinned-host memory is disabled, available pinned-host memory is always reported
     * as zero regardless of the configured limit.
     *
     * @param device_mr Device memory resource used for device allocations. To ensure
     * allocations are tracked for memory-limit accounting and statistics, use
     * `BufferResource::device_mr()` instead of the original memory resource after
     * construction.
     * @param pinned_pool_properties Configuration for the pinned host memory pool
     * used for `MemoryType::PINNED_HOST` allocations, or `PinnedMemoryDisabled` to
     * disable pinned allocations. The pinned resource is constructed internally and
     * owned by the `BufferResource`. When a value is provided, pinned host memory
     * must be supported on the system (see `is_pinned_memory_resources_supported()`);
     * otherwise a `std::runtime_error` is thrown.
     * @param memory_limits Maximum allocation limits in bytes per `MemoryType`. Missing
     * entries are treated as unlimited.
     * @param periodic_spill_check Interval between periodic spill checks. `std::nullopt`
     * disables the dedicated spill-check thread.
     * @param stream_pool CUDA stream pool used for operations that do not take an
     * explicit CUDA stream.
     * @param statistics Statistics instance used for runtime metrics.
     * @param spill_directory Directory for disk files. When set, a
     * `DiskResource` is created with an exclusively owned subdirectory under this
     * path. `std::nullopt` disables disk I/O.
     * @return A newly constructed `BufferResource` owned by `std::shared_ptr`.
     * @throws std::runtime_error if `pinned_pool_properties` has a value but pinned
     * host memory is not supported on this system.
     */
    [[nodiscard]] static std::shared_ptr<BufferResource> create(
        cuda::mr::any_resource<cuda::mr::device_accessible> device_mr,
        std::optional<PinnedPoolProperties> pinned_pool_properties = PinnedMemoryDisabled,
        std::unordered_map<MemoryType, std::int64_t> memory_limits = {},
        std::optional<Duration> periodic_spill_check = std::chrono::milliseconds{1},
        std::shared_ptr<StreamPool> stream_pool = std::make_shared<StreamPool>(16),
        std::shared_ptr<Statistics> statistics = Statistics::disabled(),
        std::optional<std::filesystem::path> spill_directory = std::nullopt
    );

    /**
     * @brief Construct a BufferResource from configuration options.
     *
     * This factory method creates a BufferResource using configuration options to
     * initialize all components. The supplied device memory resource is wrapped in
     * an internal `RmmResourceAdaptor` for allocation tracking.
     *
     * @param mr A device-accessible RMM memory resource.
     * @param options Configuration options.
     * @param statistics The statistics instance to use (disabled by default).
     * @return A shared pointer to a BufferResource instance configured according to the
     * options.
     */
    static std::shared_ptr<BufferResource> from_options(
        cuda::mr::any_resource<cuda::mr::device_accessible> mr,
        config::Options options,
        std::shared_ptr<Statistics> statistics = Statistics::disabled()
    );

    ~BufferResource() noexcept = default;

    /**
     * @brief `BufferResource` is non-copyable, it is owned by `std::shared_ptr`.
     */
    BufferResource(BufferResource const&) = delete;
    /**
     * @brief `BufferResource` is non-movable, it is owned by `std::shared_ptr`.
     */
    BufferResource(BufferResource&&) = delete;
    /**
     * @brief `BufferResource` is non-copyable.
     * @return Reference to this.
     */
    BufferResource& operator=(BufferResource const&) = delete;
    /**
     * @brief `BufferResource` is non-movable.
     * @return Reference to this.
     */
    BufferResource& operator=(BufferResource&&) = delete;

    /**
     * @brief Get the device memory resource.
     *
     * @return `rmm::device_async_resource_ref` to the device memory resource.
     *
     * @par CCCL's lifetime semantic
     *
     * The returned `rmm::device_async_resource_ref` is a non-owning
     * `cuda::mr::resource_ref`, so callers must take care to avoid use-after-free issues.
     *
     * When working directly with the returned reference, the caller must ensure that this
     * `BufferResource` remains alive for the full duration of that use:
     * @code
     * auto br = BufferResource::create(...);
     * auto mr = br->device_mr();
     * mr.allocate_async(...);  // direct use through a non-owning ref
     * br.reset();              // do not destroy `br` while `mr` is in use
     * @endcode
     *
     * To store the resource beyond the immediate call, promote the ref to an
     * owning `cuda::mr::any_resource`:
     * @code
     * auto br = BufferResource::create(...);
     * cuda::mr::any_resource<cuda::mr::device_accessible> mr = br->device_mr();
     * br.reset();       // safe: `mr` keeps the BufferResource alive
     * mr.allocate(...); // safe
     * @endcode
     *
     * In the common case, no explicit promotion is needed because RMM containers that
     * store a memory resource do this internally:
     * @code
     * auto br = BufferResource::create(...);
     * rmm::device_buffer buf{1024, stream, br->device_mr()};
     * br.reset();  // safe: `buf` keeps the BufferResource alive internally
     * @endcode
     *
     * @note Device memory resource provided to the constructor is wrapped in an
     * `RmmResourceAdaptor` for allocation tracking, and concretely the returned
     * resource_ref points to that adaptor. See `device_mr_adaptor()` for a more
     * convenient way to access the adaptor.
     */
    [[nodiscard]] rmm::device_async_resource_ref device_mr() noexcept;

    /**
     * @brief Access the concrete device memory resource adaptor.
     *
     * `BufferResource` wraps the device memory resource in an internal
     * `RmmResourceAdaptor` for allocation tracking. This exposes that adaptor
     * directly, e.g. to query allocation statistics via `get_main_record()` or
     * `current_allocated()`.
     *
     * @return Reference to the internal device `RmmResourceAdaptor`. The
     * reference is valid for as long as this `BufferResource` is alive.
     *
     * @note To ensure that the allocations are properly tracked, use `device_mr()` or
     * `device_mr_adaptor()` instead of the original memory resource passed to the
     * constructor.
     */
    [[nodiscard]] RmmResourceAdaptor& device_mr_adaptor() noexcept;

    /**
     * @brief Get the RMM host memory resource.
     *
     * @return Reference to the RMM resource used for host allocations.
     *
     * @note Lifetime semantics are identical to `device_mr()`. See its
     * `@par CCCL lifetime semantics` section for details. In brief, the returned
     * `resource_ref` is non-owning. Promote it to a `any_host_resource` to extend the
     * `BufferResource` lifetime.
     */
    [[nodiscard]] rmm::host_async_resource_ref host_mr() noexcept;

    /**
     * @brief Get the RMM pinned host memory resource.
     *
     * @throws std::invalid_argument if no pinned memory resource is available.
     * @return Reference to the RMM resource used for pinned host allocations.
     *
     * @note Lifetime semantics are identical to `device_mr()`. See its
     * `@par CCCL lifetime semantics` section for details. In brief, the returned
     * `resource_ref` is non-owning. Promote it to a `any_host_device_resource` to extend
     * the `BufferResource` lifetime.
     */
    [[nodiscard]] rmm::host_device_async_resource_ref pinned_mr();

    /**
     * @brief Get the pinned host memory resource if available.
     *
     * @return The `PinnedMemoryResource` is available, or `std::nullopt` if pinned host
     * memory is not available. The returned handle keeps this `BufferResource` alive as
     * long as the handle (or any copy) exists.
     */
    [[nodiscard]] std::optional<PinnedMemoryResource> try_pinned_mr() const;

    /**
     * @brief Returns the currently available memory for a given memory type, in bytes.
     *
     * Computed as `limit - allocated`, where `allocated` is reported by the
     * memory type's allocation counter (see the constructor documentation for
     * how each memory type is tracked). The value may be negative when
     * allocations exceed the configured limit.
     *
     * @param mem_type The memory type to query.
     * @return The available memory in bytes.
     */
    [[nodiscard]] std::int64_t memory_available(MemoryType mem_type) const noexcept;

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
    void set_memory_limit(MemoryType mem_type, std::int64_t limit) noexcept;

    /**
     * @brief Returns the memory available to a new reservation, in bytes.
     *
     * A snapshot of `memory_available(mem_type)` minus the outstanding reservations
     * of that memory type. May be negative.
     *
     * @param mem_type The memory type to query.
     * @return The memory available for reservation in bytes.
     */
    [[nodiscard]] std::int64_t memory_available_for_reservation(
        MemoryType mem_type
    ) const;

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
     * @throws std::invalid_argument if the memory type is `MemoryType::DISK` and
     * no disk resource is available.
     */
    std::pair<MemoryReservation, std::size_t> reserve(
        MemoryType mem_type, std::size_t size, AllowOverbooking allow_overbooking
    );

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
    );

    /**
     * @brief Try to reserve memory from the given order of memory types.
     *
     * @param size The size of the memory to reserve.
     * @param mem_types Memory types to try in preference order.
     * @return A full reservation, or `std::nullopt` if none is immediately available.
     */
    template <std::ranges::input_range Range>
        requires std::convertible_to<std::ranges::range_value_t<Range>, MemoryType>
    [[nodiscard]] std::optional<MemoryReservation> try_reserve(
        std::size_t size, Range mem_types
    ) {
        for (auto const& mem_type : mem_types) {
            if (mem_type == MemoryType::DISK && disk_resource_ == nullptr) {
                continue;
            }
            if (mem_type == MemoryType::PINNED_HOST && !pinned_mr_.has_value()) {
                // Pinned host memory is only available if the memory resource is
                // available.
                continue;
            }
            auto [res, _] = reserve(mem_type, size, AllowOverbooking::NO);
            if (res.size() == size) {
                return std::move(res);
            }
        }
        return std::nullopt;
    }

    /**
     * @brief Try to reserve one memory type.
     *
     * @param size The size of the memory to reserve.
     * @param mem_type Memory type to reserve.
     * @return A full reservation, or `std::nullopt` if unavailable.
     */
    [[nodiscard]] std::optional<MemoryReservation> try_reserve(
        std::size_t size, MemoryType mem_type
    ) {
        return try_reserve(size, std::ranges::single_view{mem_type});
    }

    /**
     * @brief Try to reserve memory, spilling device memory when necessary.
     *
     * Tries the requested memory types in preference order. A device reservation is
     * retained while lower memory tiers are tried; if none are immediately available,
     * device memory is spilled and retried up to @p num_spill_retries times.
     *
     * @param size The size of the memory to reserve.
     * @param mem_types Memory types to try in preference order.
     * @param num_spill_retries Maximum number of device spill attempts.
     * @return A full reservation, or `std::nullopt` if no reservation can be satisfied.
     * @throws std::invalid_argument if @p mem_types is empty.
     */
    template <std::ranges::input_range Range>
        requires std::convertible_to<std::ranges::range_value_t<Range>, MemoryType>
    [[nodiscard]] std::optional<MemoryReservation> try_reserve_or_spill(
        std::size_t size, Range mem_types, std::size_t num_spill_retries = 8
    ) {
        auto first = std::ranges::begin(mem_types);
        auto const last = std::ranges::end(mem_types);
        RAPIDSMPF_EXPECTS(
            first != last, "mem_types cannot be empty", std::invalid_argument
        );

        if (size == 0) {
            auto const mem_type = static_cast<MemoryType>(*first);
            return MemoryReservation{mem_type, this, 0};
        }

        std::optional<MemoryReservation> device_reservation;
        std::size_t device_overbooking{0};
        std::array<bool, MEMORY_TYPES.size()> seen{};

        for (; first != last; ++first) {
            auto const mem_type = static_cast<MemoryType>(*first);
            auto const index = static_cast<std::size_t>(mem_type);
            RAPIDSMPF_EXPECTS(index < seen.size(), "invalid memory type");
            if (std::exchange(seen[index], true)) {
                continue;
            }
            if (mem_type == MemoryType::DEVICE) {
                auto [reservation, overbooking] =
                    reserve(mem_type, size, AllowOverbooking::YES);
                if (overbooking == 0) {
                    return std::move(reservation);
                }
                device_overbooking = overbooking;
                device_reservation.emplace(std::move(reservation));
            } else if (auto reservation = try_reserve(size, mem_type)) {
                return reservation;
            }
        }

        if (!device_reservation.has_value()) {
            return std::nullopt;
        }
        for (std::size_t attempt = 0; attempt < num_spill_retries; ++attempt) {
            auto const spilled = spill_manager_.spill(device_overbooking);
            if (spilled >= device_overbooking) {
                return device_reservation;
            }
            device_overbooking -= spilled;
        }
        return std::nullopt;
    }

    /**
     * @brief Try to reserve one memory type, spilling device memory when necessary.
     *
     * @param size The size of the memory to reserve.
     * @param mem_type Memory type to reserve.
     * @param num_spill_retries Maximum number of device spill attempts.
     * @return A full reservation, or `std::nullopt` if unavailable.
     */
    [[nodiscard]] std::optional<MemoryReservation> try_reserve_or_spill(
        std::size_t size, MemoryType mem_type, std::size_t num_spill_retries = 8
    ) {
        return try_reserve_or_spill(
            size, std::ranges::single_view{mem_type}, num_spill_retries
        );
    }

    /**
     * @brief Make a memory reservation or fail based on the given order of memory types.
     *
     * @param size The size of the buffer to allocate.
     * @param mem_types Range of memory types to try in preference order.
     * @return A memory reservation.
     * @throws std::runtime_error if no memory reservation was made.
     */
    template <std::ranges::input_range Range>
        requires std::convertible_to<std::ranges::range_value_t<Range>, MemoryType>
    [[nodiscard]] MemoryReservation reserve_or_fail(std::size_t size, Range mem_types) {
        if (auto reservation = try_reserve(size, mem_types)) {
            return std::move(*reservation);
        }
        RAPIDSMPF_FAIL("failed to reserve memory", std::runtime_error);
    }

    /**
     * @brief Make a memory reservation or fail.
     *
     * @param size The size of the buffer to allocate.
     * @param mem_type The memory type to try to reserve memory from.
     * @return A memory reservation.
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
    std::size_t release(MemoryReservation& reservation, std::size_t size);

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
        std::size_t size, cuda::stream_ref stream, MemoryReservation& reservation
    );

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
        cuda::stream_ref stream, MemoryReservation&& reservation
    );

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
     * @param stream CUDA stream associated with the new Buffer. Use or synchronize with
     * this stream when operating on the Buffer.
     * @param spill_token The spill token the new Buffer adopts, for a caller that
     * spilled the data. Must be null for device memory.
     * @return Unique pointer to the resulting Buffer.
     *
     * @throws std::invalid_argument If @p spill_token is set and @p data is not
     * host-accessible.
     */
    std::unique_ptr<Buffer> move(
        std::unique_ptr<rmm::device_buffer> data,
        cuda::stream_ref stream,
        std::shared_ptr<SpillTrackToken> spill_token = nullptr
    );

    /**
     * @brief Move a Buffer to the memory type specified by the reservation.
     *
     * If the Buffer already resides in the target memory type, a cheap move is performed.
     * Otherwise, the Buffer is copied to the target memory using its own CUDA stream.
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
    );

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
     * @throws std::invalid_argument If the reservation's memory type isn't device memory.
     * @throws rapidsmpf::reservation_error if the memory requirement exceeds the
     * reservation.
     */
    std::unique_ptr<rmm::device_buffer> move_to_device_buffer(
        std::unique_ptr<Buffer> buffer, MemoryReservation& reservation
    );

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
     * @throws std::invalid_argument If the reservation's memory type isn't host memory.
     * @throws rapidsmpf::reservation_error If the allocation size exceeds the
     * reservation.
     */
    std::unique_ptr<HostBuffer> move_to_host_buffer(
        std::unique_ptr<Buffer> buffer, MemoryReservation& reservation
    );

    /**
     * @brief Returns the CUDA stream pool used by this buffer resource.
     *
     * Use this pool for operations that do not take an explicit CUDA stream.
     *
     * @return Shared pointer to the CUDA stream pool.
     */
    std::shared_ptr<StreamPool> const& stream_pool() const;

    /**
     * @brief Gets a reference to the spill manager used.
     *
     * @return Reference to the SpillManager instance.
     */
    SpillManager& spill_manager();

    /**
     * @brief Gets a shared pointer to the statistics associated with this buffer
     * resource.
     *
     * @return Shared pointer the Statistics instance.
     */
    std::shared_ptr<Statistics> statistics() const noexcept;

    /**
     * @brief Disk I/O resource and spill directory configuration.
     *
     * @return Shared pointer to the disk resource, or `nullptr` if no spill
     * directory was configured.
     */
    [[nodiscard]] constexpr std::shared_ptr<DiskResource> const& disk_resource() const {
        return disk_resource_;
    }

  private:
    /** @brief Private constructor, use `create()` or `from_options()`. */
    BufferResource(
        cuda::mr::any_resource<cuda::mr::device_accessible> device_mr,
        std::optional<PinnedMemoryResource> pinned_mr,
        std::unordered_map<MemoryType, std::int64_t> memory_limits,
        std::optional<Duration> periodic_spill_check,
        std::shared_ptr<StreamPool> stream_pool,
        std::shared_ptr<Statistics> statistics,
        std::shared_ptr<DiskResource> disk_resource
    );

    mutable std::mutex mutex_;
    RmmResourceAdaptor owning_mr_;
    std::optional<PinnedMemoryResource> pinned_mr_;
    HostMemoryResource host_mr_;
    std::shared_ptr<DiskResource> disk_resource_;
    std::array<std::atomic<std::int64_t>, MEMORY_TYPES.size()> memory_limits_;
    // Zero initialized reserved counters.
    std::array<std::size_t, MEMORY_TYPES.size()> memory_reserved_ = {};
    std::shared_ptr<StreamPool> stream_pool_;
    SpillManager spill_manager_;
    std::shared_ptr<Statistics> statistics_;
};

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
 * @brief Parse the `spill_host_limit` parameter from configuration options.
 *
 * Reads the `spill_host_limit` option: a byte count (e.g. `"96GiB"`) or a
 * percentage of total physical host memory (e.g. `"10%"`). When set,
 * `BufferResource::from_options` caps the `MemoryType::HOST` tier at this
 * value, so a `{HOST, DISK}` spill order becomes a *bounded* host-RAM tier in
 * front of disk. Unset (the default) leaves the host tier unlimited, as before.
 *
 * @param options Configuration options.
 *
 * @return The host memory limit in bytes, or std::nullopt when unset.
 */
std::optional<std::int64_t> host_limit_from_options(config::Options options);

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
std::shared_ptr<StreamPool> stream_pool_from_options(config::Options options);


}  // namespace rapidsmpf
