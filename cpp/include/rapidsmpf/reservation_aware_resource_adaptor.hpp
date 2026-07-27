/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda/memory_resource>

#include <rmm/resource_ref.hpp>

#include <rapidsmpf/detail/reservation_aware_resource_adaptor_impl.hpp>
#include <rapidsmpf/memory/resource_types.hpp>
#include <rapidsmpf/memory/scoped_memory_record.hpp>
#include <rapidsmpf/utils/misc.hpp>

namespace rapidsmpf::experimental {

class MemoryReservation;

/**
 * @brief Policy controlling whether a reservation may exceed the adaptor's limit.
 *
 * Distinct from `rapidsmpf::AllowOverbooking`, which governs `BufferResource`
 * reservations.
 */
enum class AllowOverbooking : bool {
    NO,  ///< Fail the request rather than exceed the limit.
    YES,  ///< Grant the request even when the memory isn't available.
};

/**
 * @brief A memory resource adaptor that only allocates through reservations.
 *
 * This adaptor wraps a primary device memory resource and adds a memory limit on top
 * of the allocation tracking provided by `RmmResourceAdaptor`. Memory is obtained by
 * calling `reserve()` and allocating through the returned `MemoryReservation`.
 *
 * Like `RmmResourceAdaptor`, this class is copyable and shares ownership of its
 * internal state via `cuda::mr::shared_resource`.
 *
 * @par Allocating without a reservation
 *
 * The adaptor is itself a memory resource, so it can be handed to cudf or an RMM
 * container directly. Those allocations are tracked, and therefore still consume
 * `available()`, but they draw on no reservation and are not capped. Allocate through
 * a `MemoryReservation` when the budget has to be enforced.
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
  public:
    /// @brief The adaptor's shared implementation.
    using Impl = detail::ReservationAwareResourceAdaptorImpl<any_device_resource>;

    /// @brief The reference-counted handle on the shared implementation.
    using shared_base = cuda::mr::shared_resource<Impl>;

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
     * memory isn't available. In that case the caller must free (at least)
     * `MemoryReservation::overbooking()` bytes before using the reservation.
     *
     * If overbooking isn't allowed, a reservation of size zero is returned on failure,
     * with `MemoryReservation::overbooking()` reporting by how much the request missed.
     * Note that unlike `BufferResource::reserve()`, a zero-sized reservation here
     * fails at allocation time rather than at buffer-construction time: the first
     * allocation through it throws `rmm::out_of_memory`.
     *
     * @param size The number of bytes to reserve.
     * @param allow_overbooking Whether overbooking is allowed.
     * @return The reservation. On success its grant always equals @p size and on
     * failure it always equals zero (a zero-sized reservation never fails).
     */
    [[nodiscard]] MemoryReservation reserve(
        std::size_t size, AllowOverbooking allow_overbooking
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

/**
 * @brief A memory reservation that is itself a memory resource.
 *
 * Granted by `ReservationAwareResourceAdaptor::reserve()`, a reservation holds a budget
 * of bytes carved out of the adaptor's limit. It is an RMM memory resource, so it can
 * be handed to cudf (or anything else taking a `rmm::device_async_resource_ref`), and
 * every allocation made through it is charged against that budget. An allocation
 * exceeding the remaining `balance()` throws `rmm::out_of_memory`; deallocating returns
 * the bytes to the balance.
 *
 * @par Ownership
 *
 * Like `ReservationAwareResourceAdaptor`, this is a `cuda::mr::shared_resource`, so
 * copies share the same reservation and are interchangeable. RMM stores such a copy
 * inside every buffer allocated from the reservation, which is what keeps the
 * reservation alive for as long as those buffers need it to service deallocations.
 *
 * The unspent balance is refunded to the adaptor when the last copy dies. Reserving
 * more than is allocated therefore keeps the surplus out of circulation for as long as
 * any derived buffer lives, so reserve what you actually use.
 *
 * @code{.cpp}
 * auto res = adaptor.reserve(1 << 30, AllowOverbooking::NO);
 * auto table = cudf::groupby(..., stream, res);
 * @endcode
 */
class MemoryReservation
    : public cuda::mr::shared_resource<
          detail::MemoryReservationImpl<ReservationAwareResourceAdaptor>> {
    using shared_base = cuda::mr::shared_resource<
        detail::MemoryReservationImpl<ReservationAwareResourceAdaptor>>;

  public:
    /// @brief The shared state of the reservation.
    using Impl = detail::MemoryReservationImpl<ReservationAwareResourceAdaptor>;

    /// @brief Tag this resource as device-accessible for the CCCL concept.
    friend void get_property(
        MemoryReservation const&, cuda::mr::device_accessible
    ) noexcept {}

    /**
     * @brief Equality comparison.
     *
     * @param other The other reservation to compare.
     * @return True if both refer to the same reservation.
     */
    [[nodiscard]] bool operator==(MemoryReservation const& other) const noexcept {
        return get() == other.get();
    }

    /**
     * @brief The number of bytes originally granted.
     *
     * @return The granted size in bytes.
     */
    [[nodiscard]] std::size_t grant() const noexcept {
        return safe_cast<std::size_t>(get().grant());
    }

    /**
     * @brief The remaining unallocated size of the reservation.
     *
     * @return The remaining size in bytes.
     */
    [[nodiscard]] std::size_t balance() const noexcept {
        return safe_cast<std::size_t>(get().balance());
    }

    /**
     * @brief The number of bytes by which the grant overbooks the adaptor's limit.
     *
     * Nonzero only when the reservation was granted with `AllowOverbooking::YES`. The
     * caller must free at least this much memory before using the reservation.
     *
     * @return The overbooked size in bytes.
     */
    [[nodiscard]] std::size_t overbooking() const noexcept {
        return overbooking_;
    }

    /**
     * @brief The adaptor that granted the reservation.
     *
     * @return The adaptor.
     */
    [[nodiscard]] ReservationAwareResourceAdaptor const& adaptor() const noexcept {
        return get().adaptor();
    }

  private:
    friend class ReservationAwareResourceAdaptor;

    /**
     * @brief Construct from an already-granted reservation.
     *
     * Private so that only `ReservationAwareResourceAdaptor` can grant reservations.
     * The reservation holds a copy of the adaptor, so the adaptor stays alive for as
     * long as any buffer allocated from the reservation needs it.
     *
     * @param adaptor The adaptor that granted the reservation.
     * @param granted The number of bytes granted.
     * @param overbooking The number of bytes by which @p granted overbooks the limit.
     */
    MemoryReservation(
        ReservationAwareResourceAdaptor const& adaptor,
        std::size_t granted,
        std::size_t overbooking
    );

    std::size_t overbooking_;
};

// Being a memory resource is the whole point of the type.
static_assert(cuda::mr::resource_with<MemoryReservation, cuda::mr::device_accessible>);

}  // namespace rapidsmpf::experimental
