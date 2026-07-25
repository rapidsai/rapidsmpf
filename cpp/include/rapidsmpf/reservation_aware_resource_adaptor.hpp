/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include <cuda/memory_resource>

#include <rmm/resource_ref.hpp>

#include <rapidsmpf/detail/reservation_aware_resource_adaptor_impl.hpp>
#include <rapidsmpf/memory/resource_types.hpp>
#include <rapidsmpf/memory/scoped_memory_record.hpp>
#include <rapidsmpf/utils/misc.hpp>

namespace rapidsmpf {

class ReservationAwareResourceAdaptor;

/**
 * @brief An owning handle to a memory reservation that is itself a memory resource.
 *
 * Unlike `MemoryReservation`, which is passive bookkeeping consumed by
 * `BufferResource::make_buffer()`, this reservation *is* an RMM memory resource. It
 * can be handed to cudf (or anything else taking a
 * `rmm::device_async_resource_ref`), and every allocation made through it is charged
 * against the reservation. That makes "reserve before allocate" enforceable rather
 * than advisory: an allocation that exceeds the reservation throws
 * `rmm::out_of_memory`.
 *
 * @par Ownership
 *
 * This type is move-only and is the sole owner of the reservation. The handle
 * returned by `mr()` is copyable and shares the underlying state; RMM copies it into
 * every buffer allocated from the reservation, which keeps the state alive for as
 * long as those buffers need it to service deallocations.
 *
 * Destroying (or explicitly `release()`ing) this owner refunds the unused balance
 * immediately, at the end of the reserving scope, rather than waiting for the last
 * derived buffer to die. Buffers that outlive the owner keep working: further
 * allocations through them fall through to the adaptor, tracked but unreserved.
 *
 * @code{.cpp}
 * auto [res, overbooking] = adaptor.reserve(1 << 30, false);
 * auto table = cudf::groupby(..., stream, res.mr());
 * // `res` goes out of scope here: the unused balance is refunded immediately, while
 * // `table` keeps the reservation state alive to service its own deallocations.
 * @endcode
 */
class MemoryReservation2 {
  public:
    /// @brief The shared state of the reservation.
    using Impl = detail::ReservationImpl<any_device_resource>;

    ~MemoryReservation2() noexcept {
        release();
    }

    /**
     * @brief Move constructor.
     *
     * @param o The reservation to move from.
     */
    MemoryReservation2(MemoryReservation2&& o) noexcept
        : handle_{std::exchange(o.handle_, std::nullopt)} {}

    /**
     * @brief Move assignment operator.
     *
     * @param o The reservation to move from.
     * @return Reference to this.
     */
    MemoryReservation2& operator=(MemoryReservation2&& o) noexcept {
        if (this != std::addressof(o)) {
            release();
            handle_ = std::exchange(o.handle_, std::nullopt);
        }
        return *this;
    }

    /// @brief A memory reservation is not copyable.
    MemoryReservation2(MemoryReservation2 const&) = delete;
    /// @brief A memory reservation is not copyable.
    MemoryReservation2& operator=(MemoryReservation2 const&) = delete;

    /**
     * @brief Refund the unused balance to the adaptor.
     *
     * Idempotent. Allocations already made through this reservation stay valid, and
     * buffers holding a copy of `mr()` continue to work, but they no longer draw on a
     * reservation.
     */
    void release() noexcept {
        if (handle_.has_value()) {
            (*handle_)->release_owner();
        }
    }

    /**
     * @brief The memory resource to allocate from.
     *
     * Pass this to cudf, RMM containers, or anything else accepting a
     * `cuda::mr::any_resource` or an `rmm::device_async_resource_ref`.
     *
     * Allocations through this handle, or through any copy of it, draw on the
     * reservation's balance and are capped by it. A copy does not extend the budget's
     * lifetime, though: destroying this `MemoryReservation2` refunds the unspent balance
     * to the adaptor immediately. Copies remain usable afterwards, keeping the
     * reservation's bookkeeping alive so buffers can still deallocate, but allocations
     * through them are then tracked and unreserved.
     *
     * Returning by value is what keeps that safe, so this is deliberately not an
     * `rmm::device_async_resource_ref`.
     *
     * @return An owning handle to the reservation, usable as a memory resource.
     */
    [[nodiscard]] cuda::mr::shared_resource<Impl> mr() const noexcept {
        return *handle_;
    }

    /**
     * @brief The number of bytes originally granted.
     *
     * @return The granted size in bytes.
     */
    [[nodiscard]] std::size_t grant() const noexcept {
        return handle_.has_value() ? safe_cast<std::size_t>((*handle_)->grant()) : 0;
    }

    /**
     * @brief The remaining unallocated size of the reservation.
     *
     * @return The remaining size in bytes, or zero once released.
     */
    [[nodiscard]] std::size_t size() const noexcept {
        return handle_.has_value() ? safe_cast<std::size_t>((*handle_)->balance()) : 0;
    }

  private:
    friend class ReservationAwareResourceAdaptor;

    /**
     * @brief Construct from an already-granted reservation.
     *
     * Private so that only `ReservationAwareResourceAdaptor` can grant reservations.
     *
     * @param handle The shared reservation state.
     */
    explicit MemoryReservation2(cuda::mr::shared_resource<Impl> handle)
        : handle_{std::move(handle)} {}

    /// @brief The shared reservation state, `std::nullopt` once moved from.
    std::optional<cuda::mr::shared_resource<Impl>> handle_;
};

// What `mr()` hands out is a memory resource, which is the whole point of the type.
static_assert(cuda::mr::resource_with<
              cuda::mr::shared_resource<MemoryReservation2::Impl>,
              cuda::mr::device_accessible>);

/**
 * @brief A memory resource adaptor that only allocates through reservations.
 *
 * This adaptor wraps a primary device memory resource and adds a memory limit on top
 * of the allocation tracking provided by `RmmResourceAdaptor`. Memory is obtained by
 * calling `reserve()` and allocating through the returned `MemoryReservation2`.
 *
 * Like `RmmResourceAdaptor`, this class is copyable and shares ownership of its
 * internal state via `cuda::mr::shared_resource`.
 *
 * @par Allocating without a reservation
 *
 * The adaptor is itself a memory resource, so it can be handed to cudf or an RMM
 * container directly. Those allocations are tracked, and therefore still consume
 * `available()`, but they draw on no reservation and are not capped. Allocate through
 * a `MemoryReservation2` when the budget has to be enforced.
 *
 * @par Accounting
 *
 * Three quantities describe the state of the adaptor:
 * - `current_allocated()`: bytes currently allocated, tracked on every allocation.
 * - `total_reserved()`: bytes held by live reservations but not yet allocated.
 * - `available() == limit() - current_allocated() - total_reserved()`.
 *
 * Allocating through a reservation moves bytes from the second bucket to the first,
 * leaving `available()` unchanged; that is what makes a reservation a promise.
 */
class ReservationAwareResourceAdaptor
    : public cuda::mr::shared_resource<
          detail::ReservationAwareResourceAdaptorImpl<any_device_resource>> {
    using shared_base = cuda::mr::shared_resource<
        detail::ReservationAwareResourceAdaptorImpl<any_device_resource>>;

  public:
    /// @brief The adaptor's shared implementation.
    using Impl = detail::ReservationAwareResourceAdaptorImpl<any_device_resource>;

    /// @brief Tag this resource as device-accessible for the CCCL concept.
    friend void get_property(
        ReservationAwareResourceAdaptor const&, cuda::mr::device_accessible
    ) noexcept {}

    /**
     * @brief Construct with the specified primary memory resource and limit.
     *
     * @param primary_mr The primary memory resource.
     * @param limit Maximum number of bytes that may be allocated and reserved.
     */
    ReservationAwareResourceAdaptor(any_device_resource primary_mr, std::int64_t limit);

    /**
     * @brief Equality comparison.
     *
     * @param other The other adaptor to compare.
     * @return True if both adaptors share the same underlying state.
     */
    [[nodiscard]] bool operator==(
        ReservationAwareResourceAdaptor const& other
    ) const noexcept {
        return get() == other.get();
    }

    /**
     * @brief Reserve an amount of memory.
     *
     * Creates a new reservation of the specified size to inform about upcoming
     * allocations.
     *
     * If overbooking is allowed, a reservation of @p size is returned even when the
     * memory isn't available. In that case the caller must free (at least) the
     * overbooked amount before using the reservation.
     *
     * If overbooking isn't allowed, a reservation of size zero is returned on failure.
     * Note that unlike `BufferResource::reserve()`, a zero-sized reservation here
     * fails at allocation time rather than at buffer-construction time: the first
     * allocation through it throws `rmm::out_of_memory`.
     *
     * @param size The number of bytes to reserve.
     * @param allow_overbooking Whether overbooking is allowed.
     * @return A pair containing the reservation and the amount of overbooking. On
     * success the size of the reservation always equals @p size and on failure it
     * always equals zero (a zero-sized reservation never fails).
     */
    [[nodiscard]] std::pair<MemoryReservation2, std::size_t> reserve(
        std::size_t size, bool allow_overbooking
    );

    /**
     * @brief Get the memory limit.
     *
     * @return The limit in bytes.
     */
    [[nodiscard]] std::int64_t limit() const noexcept;

    /**
     * @brief Update the memory limit at runtime.
     *
     * @param limit The new byte limit.
     */
    void set_limit(std::int64_t limit) noexcept;

    /**
     * @brief Get the total current allocated memory through this adaptor.
     *
     * @return Total number of currently allocated bytes.
     */
    [[nodiscard]] std::int64_t current_allocated() const noexcept;

    /**
     * @brief Get the memory currently held by live reservations.
     *
     * Excludes reserved bytes that have already been allocated; those are reported by
     * `current_allocated()` instead.
     *
     * @return Total number of reserved bytes.
     */
    [[nodiscard]] std::int64_t total_reserved() const noexcept;

    /**
     * @brief Get the memory available for new reservations.
     *
     * Computed as `limit() - current_allocated() - total_reserved()`. May be negative
     * when reservations have overbooked the limit.
     *
     * @return The available memory in bytes.
     */
    [[nodiscard]] std::int64_t available() const noexcept;

    /**
     * @brief Returns a copy of the main memory record.
     *
     * @return A copy of the current main memory record.
     */
    [[nodiscard]] ScopedMemoryRecord get_main_record() const;

    /**
     * @brief Get a reference to the primary upstream resource.
     *
     * @return Reference to the RMM memory resource.
     */
    [[nodiscard]] rmm::device_async_resource_ref get_upstream_resource() const noexcept;
};

static_assert(
    cuda::mr::resource_with<ReservationAwareResourceAdaptor, cuda::mr::device_accessible>
);

}  // namespace rapidsmpf
