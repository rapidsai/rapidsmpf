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
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda/memory_resource>
#include <cuda/stream>

#include <rmm/cuda_stream_pool.hpp>

#include <rapidsmpf/disk/disk_resource.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/host_memory_resource.hpp>
#include <rapidsmpf/memory/memory_reservation.hpp>
#include <rapidsmpf/memory/memory_type.hpp>
#include <rapidsmpf/memory/pinned_memory_resource.hpp>
#include <rapidsmpf/memory/reservation_attributes.hpp>
#include <rapidsmpf/memory/reservation_events.hpp>
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

class BufferResource;

namespace streaming {
class MemoryReserveOrWait;
}

namespace detail {

/// @cond
struct ReserveResult {
    MemoryReservation reservation;
    MemoryType mem_type{MemoryType::HOST};
    std::size_t overbooked_bytes{0};
    std::size_t total_overbooked_bytes{0};
    bool waited{false};
    bool timed_out{false};
    bool cancelled{false};
};

class RequestScope {
  public:
    RequestScope(
        BufferResource& br,
        std::size_t size,
        std::span<MemoryType const> memory_types,
        ReservationRequestPolicy policy,
        std::optional<std::int64_t> net_memory_delta,
        ReservationAttributes const& attributes
    );
    ~RequestScope() noexcept;
    RequestScope(RequestScope const&) = delete;
    RequestScope& operator=(RequestScope const&) = delete;

    [[nodiscard]] std::uint64_t id() const noexcept {
        return id_;
    }

    [[nodiscard]] ReservationAttributes const& attributes() const noexcept {
        return attributes_;
    }

    [[nodiscard]] bool active() const noexcept {
        return active_;
    }

    void waiting();
    void resolve_granted(ReserveResult const&);
    void resolve_empty(bool timed_out);
    void resolve_failed(std::string const& error);
    void resolve_cancelled();

  private:
    void resolve(
        ReservationRequestOutcome outcome,
        ReserveResult const* result,
        bool timed_out,
        std::string const& error
    );

    BufferResource* br_;
    bool active_;
    std::uint64_t id_;
    ReservationAttributes attributes_;
    bool waiting_emitted_{false};
    bool resolved_{false};
};

/// @endcond

}  // namespace detail

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
     * Device, pinned-host, and pageable-host allocations routed through this
     * `BufferResource` are tracked automatically.
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
     * entries are treated as unlimited. See `memory_limits_from_options()`.
     * @param periodic_spill_check Interval between periodic spill checks. `std::nullopt`
     * disables the dedicated spill-check thread.
     * @param stream_pool CUDA stream pool used for operations that do not take an
     * explicit CUDA stream.
     * @param statistics Statistics instance used for runtime metrics.
     * @param spill_directory Directory for disk files. When set, a
     * `DiskResource` is created with an exclusively owned subdirectory under this
     * path. `std::nullopt` disables disk I/O.
     * @param memory_reserve_timeout Progress timeout for memory reservation requests.
     * See `memory_reserve_timeout_from_options()`.
     * @param allow_overbooking_by_default Overbooking policy for reservation requests
     * that don't specify one. See `allow_overbooking_by_default_from_options()`.
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
        std::optional<std::filesystem::path> spill_directory = std::nullopt,
        Duration memory_reserve_timeout = std::chrono::milliseconds{100},
        AllowOverbooking allow_overbooking_by_default = AllowOverbooking::YES
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

    ~BufferResource() noexcept;

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
     * `BufferResource` lifetime. To obtain an owning `HostMemoryResource` handle, use
     * `cuda::mr::resource_cast<rapidsmpf::HostMemoryResource>` on the returned ref and
     * copy the result.
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
     * @brief Return the configured limit for a memory type.
     *
     * @param mem_type The memory type to query.
     * @return The exact signed configured byte limit.
     */
    [[nodiscard]] std::int64_t memory_limit(MemoryType mem_type) const noexcept;

    /**
     * @brief Test whether this buffer resource owns a given memory resource.
     *
     * @param mem_type The memory type to query.
     * @return Whether reservations can target this memory type.
     */
    [[nodiscard]] bool has_memory_resource(MemoryType mem_type) const noexcept;

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
     * @brief Start a new reservation event recording session.
     *
     * Recording is disabled by default. A session starts with sequence zero.
     *
     * @throws std::logic_error if recording is already active.
     */
    void start_reservation_event_recording();

    /**
     * @brief Atomically detach and return the events recorded so far.
     *
     * Recording remains active. Returns an empty batch if recording is disabled
     * or no events have occurred since the previous drain.
     *
     * @returns The events recorded since the previous drain.
     */
    [[nodiscard]] ReservationEventBatch drain_reservation_events();

    /**
     * @brief Disable recording and atomically return the final event batch.
     *
     * Once this method returns, no producer can append to the stopped session.
     * Calling it while recording is disabled returns an empty batch.
     *
     * @returns The final events recorded by the stopped session.
     */
    [[nodiscard]] ReservationEventBatch stop_reservation_event_recording();

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
     * @brief Reserve memory and attach immutable telemetry attributes.
     *
     * @param mem_type The target memory type.
     * @param size The number of bytes to reserve.
     * @param allow_overbooking Whether overbooking is allowed.
     * @param attributes Immutable telemetry attributes for the request.
     * @return The reservation and this request's overbooked byte count.
     */
    std::pair<MemoryReservation, std::size_t> reserve(
        MemoryType mem_type,
        std::size_t size,
        AllowOverbooking allow_overbooking,
        ReservationAttributes attributes
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
     * @brief Try to make a memory reservation based on the given order of memory types.
     *
     * The function attempts to reserve memory by iterating over @p mem_types in the given
     * order of preference. For each memory type, it requests a reservation without
     * overbooking. Memory types without a backing resource are skipped.
     *
     * @param size The size of the buffer to allocate.
     * @param mem_types Range of memory types to try to reserve memory from.
     * @param attributes Immutable telemetry attributes for the request.
     * @return A memory reservation, or `std::nullopt` if no memory type can satisfy the
     * request.
     */
    template <std::ranges::input_range Range>
        requires std::convertible_to<std::ranges::range_value_t<Range>, MemoryType>
    [[nodiscard]] std::optional<MemoryReservation> try_reserve(
        std::size_t size, Range mem_types, ReservationAttributes const& attributes = {}
    ) {
        if (size == 0) {
            for (auto const mem_type : mem_types) {
                if (has_memory_resource(mem_type)) {
                    return MemoryReservation(mem_type, this, 0);
                }
            }
            return std::nullopt;
        }
        std::vector<MemoryType> type_storage;
        std::span<MemoryType const> types;
        if constexpr (std::ranges::contiguous_range<Range>
                      && std::ranges::sized_range<Range>)
        {
            types = {std::ranges::data(mem_types), std::ranges::size(mem_types)};
        } else {
            type_storage.assign(
                std::ranges::begin(mem_types), std::ranges::end(mem_types)
            );
            types = type_storage;
        }
        detail::RequestScope scope{
            *this,
            size,
            types,
            ReservationRequestPolicy::IMMEDIATE,
            std::nullopt,
            attributes
        };
        try {
            for (auto const mem_type : types) {
                if (!has_memory_resource(mem_type)) {
                    continue;
                }
                auto result = reserve_impl(mem_type, size, AllowOverbooking::NO, &scope);
                if (result.reservation.size() == size) {
                    scope.resolve_granted(result);
                    return std::move(result.reservation);
                }
            }
            scope.resolve_empty(false);
            return std::nullopt;
        } catch (std::exception const& e) {
            scope.resolve_failed(e.what());
            throw;
        } catch (...) {
            scope.resolve_failed("unknown exception");
            throw;
        }
    }

    /**
     * @brief Try to make a memory reservation.
     *
     * @param size The size of the buffer to allocate.
     * @param mem_type The memory type to try to reserve memory from.
     * @param attributes Immutable telemetry attributes for the request.
     * @return A memory reservation, or `std::nullopt` if the request cannot be
     * satisfied.
     */
    [[nodiscard]] std::optional<MemoryReservation> try_reserve(
        std::size_t size,
        MemoryType mem_type,
        ReservationAttributes const& attributes = {}
    ) {
        return try_reserve(size, std::ranges::single_view{mem_type}, attributes);
    }

    /**
     * @brief Reserve device memory, spill as needed, and attach telemetry attributes.
     *
     * @param size The size of memory to reserve.
     * @param allow_overbooking Whether overbooking is allowed.
     * @param attributes Immutable telemetry attributes for the request.
     * @return The memory reservation.
     */
    MemoryReservation reserve_device_memory_and_spill(
        std::size_t size,
        AllowOverbooking allow_overbooking,
        ReservationAttributes attributes
    );

    /**
     * @brief Make a memory reservation or fail based on the given order of memory types.
     *
     * The function attempts to reserve memory by iterating over @p mem_types in the given
     * order of preference. For each memory type, it requests a reservation without
     * overbooking. If no memory type can satisfy the request, the function throws.
     *
     * @param size The size of the buffer to allocate.
     * @param mem_types Range of memory types to try to reserve memory from.
     * @param attributes Immutable telemetry attributes for the request.
     * @return A memory reservation.
     *
     * @throws std::runtime_error if no memory reservation was made.
     */
    template <std::ranges::input_range Range>
        requires std::convertible_to<std::ranges::range_value_t<Range>, MemoryType>
    [[nodiscard]] MemoryReservation reserve_or_fail(
        std::size_t size, Range mem_types, ReservationAttributes const& attributes = {}
    ) {
        if (size == 0) {
            for (auto const mem_type : mem_types) {
                if (has_memory_resource(mem_type)) {
                    return MemoryReservation(mem_type, this, 0);
                }
            }
            RAPIDSMPF_FAIL("failed to reserve memory", std::runtime_error);
        }
        std::vector<MemoryType> type_storage;
        std::span<MemoryType const> types;
        if constexpr (std::ranges::contiguous_range<Range>
                      && std::ranges::sized_range<Range>)
        {
            types = {std::ranges::data(mem_types), std::ranges::size(mem_types)};
        } else {
            type_storage.assign(
                std::ranges::begin(mem_types), std::ranges::end(mem_types)
            );
            types = type_storage;
        }
        detail::RequestScope scope{
            *this,
            size,
            types,
            ReservationRequestPolicy::IMMEDIATE,
            std::nullopt,
            attributes
        };
        try {
            for (auto const mem_type : types) {
                if (!has_memory_resource(mem_type)) {
                    continue;
                }
                auto result = reserve_impl(mem_type, size, AllowOverbooking::NO, &scope);
                if (result.reservation.size() == size) {
                    scope.resolve_granted(result);
                    return std::move(result.reservation);
                }
            }
            throw std::runtime_error("failed to reserve memory");
        } catch (std::exception const& e) {
            scope.resolve_failed(e.what());
            throw;
        } catch (...) {
            scope.resolve_failed("unknown exception");
            throw;
        }
    }

    /**
     * @brief Make a memory reservation or fail.
     *
     * @param size The size of the buffer to allocate.
     * @param mem_type The memory type to try to reserve memory from.
     * @param attributes Immutable telemetry attributes for the request.
     * @return A memory reservation.
     *
     * @throws std::runtime_error if no memory reservation was made.
     */
    [[nodiscard]] MemoryReservation reserve_or_fail(
        std::size_t size,
        MemoryType mem_type,
        ReservationAttributes const& attributes = {}
    ) {
        return reserve_or_fail(size, std::ranges::single_view{mem_type}, attributes);
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
     * @throws std::invalid_argument if @p stream is `cuda::invalid_stream`.
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
     *
     * @throws std::invalid_argument if @p stream is `cuda::invalid_stream`.
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
     * @p stream is ordered after any work pending on the device buffer's current
     * stream, and the device buffer's stream is updated to @p stream. If @p data is
     * empty (e.g. a default-constructed `rmm::device_buffer`, which has no stream), it
     * owns no memory, so no synchronization is performed.
     *
     * @param data Unique pointer to the device or pinned host buffer.
     * @param stream CUDA stream associated with the new Buffer. Use or synchronize with
     * this stream when operating on the Buffer.
     * @param spill_token The spill token the new Buffer adopts, for a caller that
     * spilled the data. Must be null for device memory.
     * @return Unique pointer to the resulting Buffer.
     *
     * @throws std::invalid_argument If @p spill_token is set and @p data is not
     * host-accessible, or if @p stream is `cuda::invalid_stream`.
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
     * @brief Move a buffer into one of @p accepted, unless it is already in one of them.
     *
     * The memory types are tried in reverse order of @p accepted without overbooking.
     * If none of them fits, the last memory type is overbooked if
     * `allow_overbooking_by_default()` allows it. The move may block the calling
     * thread.
     *
     * @param buffer The buffer. Must not be `nullptr`.
     * @param accepted The accepted memory types in decreasing order of preference.
     * Must not be empty.
     * @return @p buffer, unchanged if it is already in one of @p accepted, otherwise
     * moved into one of them.
     *
     * @throws std::invalid_argument If @p buffer is `nullptr` or @p accepted is empty.
     * @throws rapidsmpf::reservation_error If @p buffer must be moved, none of
     * @p accepted fits, and overbooking is not allowed.
     */
    [[nodiscard]] std::unique_ptr<Buffer> ensure_memory_type(
        std::unique_ptr<Buffer> buffer, std::span<MemoryType const> accepted
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

    /**
     * @brief Progress timeout for memory reservation requests.
     *
     * The longest time a component waiting for memory may go without any of its
     * pending reservations being satisfied before it forces progress. See the
     * `memory_reserve_timeout` option.
     *
     * @return The timeout.
     */
    [[nodiscard]] Duration memory_reserve_timeout() const noexcept {
        return memory_reserve_timeout_;
    }

    /**
     * @brief Overbooking policy for reservation requests that don't specify one.
     *
     * Decides whether forcing progress after `memory_reserve_timeout()` may overbook
     * memory. See the `allow_overbooking_by_default` option.
     *
     * @return The overbooking policy.
     */
    [[nodiscard]] AllowOverbooking allow_overbooking_by_default() const noexcept {
        return allow_overbooking_by_default_;
    }

  private:
    friend class MemoryReservation;
    friend class detail::RequestScope;
    friend class streaming::MemoryReserveOrWait;

    [[nodiscard]] bool reservation_event_recording_enabled() const noexcept {
        return recording_enabled_.load(std::memory_order_acquire);
    }

    detail::ReserveResult reserve_impl(
        MemoryType mem_type,
        std::size_t size,
        AllowOverbooking allow_overbooking,
        detail::RequestScope* request
    );
    std::uint64_t next_reservation_id() noexcept;
    std::uint64_t next_request_id() noexcept;
    void emit_request_submitted(
        std::uint64_t,
        std::size_t,
        std::span<MemoryType const>,
        ReservationRequestPolicy,
        std::optional<std::int64_t>,
        ReservationAttributes const&
    );
    void emit_request_waiting(std::uint64_t);
    void emit_request_resolved(
        std::uint64_t,
        ReservationRequestOutcome,
        detail::ReserveResult const*,
        bool,
        std::string const&
    );
    void emit_reservation_created(
        MemoryReservation const&, std::uint64_t request_id, std::uint64_t parent_id
    );
    void emit_reservation_resized(std::uint64_t, std::size_t, ReservationResizeReason);
    void emit_reservation_released(std::uint64_t, ReservationReleaseReason);
    void record_reservation_event(ReservationEventData) noexcept;
    void clear_reservation(MemoryReservation&) noexcept;
    /** @brief Private constructor, use `create()` or `from_options()`. */
    BufferResource(
        cuda::mr::any_resource<cuda::mr::device_accessible> device_mr,
        std::optional<PinnedMemoryResource> pinned_mr,
        std::unordered_map<MemoryType, std::int64_t> memory_limits,
        std::optional<Duration> periodic_spill_check,
        std::shared_ptr<StreamPool> stream_pool,
        std::shared_ptr<Statistics> statistics,
        std::shared_ptr<DiskResource> disk_resource,
        Duration memory_reserve_timeout,
        AllowOverbooking allow_overbooking_by_default
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
    Duration memory_reserve_timeout_;
    AllowOverbooking allow_overbooking_by_default_;

    struct RecordingSession {
        ReservationEventBatch events;
        std::uint64_t next_sequence{0};
    };

    std::optional<RecordingSession> recording_;
    std::mutex recording_mutex_;
    std::atomic<bool> recording_enabled_{false};
    std::atomic<std::uint64_t> reservation_id_counter_{1};
    std::atomic<std::uint64_t> request_id_counter_{1};
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
 * @brief Build the per-`MemoryType` memory limits from configuration options.
 *
 * @param options Configuration options.
 * @param pinned_pool_properties Pinned pool configuration, typically from
 * `pinned_pool_properties_from_options()`, or `PinnedMemoryDisabled`.
 * @return Memory limits in bytes, suitable for `BufferResource::create()`.
 */
std::unordered_map<MemoryType, std::int64_t> memory_limits_from_options(
    config::Options options,
    std::optional<PinnedPoolProperties> const& pinned_pool_properties
);

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

/**
 * @brief Get the `memory_reserve_timeout` parameter from configuration options.
 *
 * @param options Configuration options.
 * @return The progress timeout for memory reservation requests.
 */
Duration memory_reserve_timeout_from_options(config::Options options);

/**
 * @brief Get the `allow_overbooking_by_default` parameter from configuration options.
 *
 * @param options Configuration options.
 * @return The overbooking policy for reservation requests that don't specify one.
 */
AllowOverbooking allow_overbooking_by_default_from_options(config::Options options);


}  // namespace rapidsmpf
