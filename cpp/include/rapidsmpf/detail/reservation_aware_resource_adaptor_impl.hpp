/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include <cuda/memory_resource>

#include <rmm/aligned.hpp>
#include <rmm/error.hpp>

#include <rapidsmpf/detail/rmm_resource_adaptor_impl.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/utils/misc.hpp>
#include <rapidsmpf/utils/string.hpp>

namespace rapidsmpf::experimental {

class ReservationAwareResourceAdaptor;

}  // namespace rapidsmpf::experimental

namespace rapidsmpf::detail {

// The adaptor is a template parameter only so that the reservation can hold one by
// value: `ReservationAwareResourceAdaptor` is defined in terms of the adaptor impl
// below, so it is still incomplete here, and only a dependent member type defers the
// completeness requirement to instantiation time.
template <typename Adaptor>
    requires std::same_as<Adaptor, experimental::ReservationAwareResourceAdaptor>
class MemoryReservationImpl;

/**
 * @brief Shared state of a ReservationAwareResourceAdaptor.
 *
 * @tparam PrimaryMR The type of the primary memory resource.
 */
template <cuda::mr::resource_with<cuda::mr::device_accessible> PrimaryMR>
class ReservationAwareResourceAdaptorImpl : public RmmResourceAdaptorImpl<PrimaryMR> {
  public:
    using Base = RmmResourceAdaptorImpl<PrimaryMR>;  ///< The tracking base class.

    /**
     * @brief Construct with a primary memory resource and a memory limit.
     *
     * @param primary_mr The primary memory resource (moved in).
     * @param limit Maximum number of bytes that may be allocated and reserved.
     */
    ReservationAwareResourceAdaptorImpl(PrimaryMR primary_mr, std::int64_t limit)
        : Base{std::move(primary_mr)}, limit_{limit} {}

    /**
     * @brief Construct the primary resource in-place from forwarded arguments.
     *
     * @param limit Maximum number of bytes that may be allocated and reserved.
     * @param tag Disambiguation tag for in-place construction.
     * @param args Arguments forwarded to the `PrimaryMR` constructor.
     */
    template <typename... Args>
    ReservationAwareResourceAdaptorImpl(
        std::int64_t limit, std::in_place_t tag, Args&&... args
    )
        : Base{tag, std::forward<Args>(args)...}, limit_{limit} {}

    ~ReservationAwareResourceAdaptorImpl() = default;

    /// @copydoc rapidsmpf::experimental::ReservationAwareResourceAdaptor::limit
    [[nodiscard]] std::int64_t limit() const noexcept {
        return limit_.load(std::memory_order_acquire);
    }

    /// @copydoc rapidsmpf::experimental::ReservationAwareResourceAdaptor::set_limit
    void set_limit(std::int64_t limit) noexcept {
        limit_.store(limit, std::memory_order_release);
    }

    /// @copydoc rapidsmpf::experimental::ReservationAwareResourceAdaptor::total_reserved
    [[nodiscard]] std::int64_t total_reserved() const noexcept {
        return total_reserved_.load(std::memory_order_acquire);
    }

    /// @copydoc rapidsmpf::experimental::ReservationAwareResourceAdaptor::available
    [[nodiscard]] std::int64_t available() const noexcept {
        return limit() - this->current_allocated() - total_reserved();
    }

    /**
     * @brief Reserve @p size bytes against the limit.
     *
     * @param size The number of bytes to reserve.
     * @param allow_overbooking Whether to grant the reservation even when the memory
     * isn't available.
     * @return A pair of the number of bytes granted (either @p size or zero) and the
     * number of bytes by which the request overbooks the limit.
     */
    [[nodiscard]] std::pair<std::size_t, std::size_t> try_reserve(
        std::size_t size, bool allow_overbooking
    ) {
        auto const want = safe_cast<std::int64_t>(size);
        auto reserved = total_reserved_.load(std::memory_order_relaxed);
        while (true) {
            // Availability *after* the reservation would be made. Negative means the
            // request overbooks the limit.
            std::int64_t const headroom =
                limit() - this->current_allocated() - (reserved + want);
            std::size_t const overbooking =
                headroom < 0 ? safe_cast<std::size_t>(-headroom) : 0;
            if (overbooking > 0 && !allow_overbooking) {
                return {0, overbooking};
            }
            if (total_reserved_.compare_exchange_weak(
                    reserved,
                    reserved + want,
                    std::memory_order_acq_rel,
                    std::memory_order_relaxed
                ))
            {
                return {size, overbooking};
            }
        }
    }

  private:
    friend class MemoryReservationImpl<experimental::ReservationAwareResourceAdaptor>;

    std::atomic<std::int64_t> limit_;
    // Reservations move bytes in and out of this counter as they allocate, free, and die.
    std::atomic<std::int64_t> total_reserved_{0};
};

/**
 * @brief Shared state of a memory reservation.
 *
 * Satisfies the `cuda::mr::resource` concept, so that it can be made a
 * `cuda::mr::shared_resource`.
 *
 * The reservation is a hard cap: an allocation exceeding the remaining balance throws.
 * Allocating moves bytes from the adaptor's reserved counter to its allocated counter
 * and deallocating moves them back, so the unspent balance goes back to the adaptor
 * only when the last reference to this state dies. Reserving more than is actually
 * allocated therefore keeps the surplus out of circulation for the whole lifetime of
 * the buffers allocated from it; over-reservation is a caller bug.
 *
 * @tparam Adaptor Always `ReservationAwareResourceAdaptor`; see the forward
 * declaration above for why it is a template parameter at all.
 */
template <typename Adaptor>
    requires std::same_as<Adaptor, experimental::ReservationAwareResourceAdaptor>
class MemoryReservationImpl {
  public:
    /**
     * @brief Construct a reservation over an already-granted number of bytes.
     *
     * @param adaptor The adaptor that granted the reservation. Held by value, so the
     * adaptor outlives every buffer allocated from the reservation.
     * @param grant The number of bytes granted.
     * @param overbooking The number of bytes by which @p grant overbooks the limit.
     */
    MemoryReservationImpl(Adaptor adaptor, std::int64_t grant, std::size_t overbooking)
        : adaptor_{std::move(adaptor)},
          grant_{grant},
          overbooking_{overbooking},
          balance_{grant} {}

    /// @brief Refund the unspent balance to the adaptor.
    ~MemoryReservationImpl() {
        adaptor_->total_reserved_.fetch_sub(balance(), std::memory_order_acq_rel);
    }

    /**
     * @brief The number of bytes originally granted.
     *
     * @return The granted size in bytes.
     */
    [[nodiscard]] std::int64_t grant() const noexcept {
        return grant_;
    }

    /**
     * @brief The number of bytes by which the grant overbooks the adaptor's limit.
     *
     * @return The overbooked size in bytes.
     */
    [[nodiscard]] std::size_t overbooking() const noexcept {
        return overbooking_;
    }

    /**
     * @brief The remaining balance.
     *
     * @return The number of granted but not yet allocated bytes.
     */
    [[nodiscard]] std::int64_t balance() const noexcept {
        return balance_.load(std::memory_order_acquire);
    }

    /**
     * @brief Allocate memory asynchronously on the given stream.
     *
     * The balance is drawn down before allocating, which is what enforces the cap, while
     * the adaptor's reserved counter is only decremented once the base has recorded the
     * allocation, so a concurrent `available()` never over-reports.
     *
     * The reservation is charged @p bytes rather than the alignment-padded size, which
     * is what the adaptor's allocation counter records. Charging the padded size instead
     * would silently consume more of the grant than the caller asked for and leave
     * `available()` reading high by the difference.
     *
     * @param stream The CUDA stream for the allocation.
     * @param bytes Number of bytes to allocate.
     * @param alignment Alignment requirement.
     * @return Pointer to the allocated memory.
     *
     * @throws rmm::out_of_memory if @p bytes exceeds the remaining balance.
     */
    void* allocate(
        cuda::stream_ref stream,
        std::size_t bytes,
        std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT
    ) {
        auto const amount = safe_cast<std::int64_t>(bytes);
        draw_down_res(amount);
        void* ptr = nullptr;
        try {
            ptr = adaptor_->allocate(stream, bytes, alignment);
        } catch (...) {
            // The allocation never happened.
            balance_.fetch_add(amount, std::memory_order_acq_rel);
            throw;
        }
        adaptor_->total_reserved_.fetch_sub(amount, std::memory_order_acq_rel);
        return ptr;
    }

    /**
     * @brief Deallocate memory asynchronously on the given stream.
     *
     * Mirrors `allocate()`: the bytes go back to the balance and the adaptor's reserved
     * counter is restored before the base drops the allocation, so a concurrent
     * `available()` never over-reports.
     *
     * @param stream The CUDA stream for the deallocation.
     * @param ptr Pointer to the memory to deallocate.
     * @param bytes Number of bytes to deallocate.
     * @param alignment Alignment of the original allocation.
     *
     * @warning As with any memory resource, @p ptr must have been allocated by this
     * same reservation. Freeing memory allocated elsewhere inflates the balance beyond
     * the grant, letting the reservation allocate more than it was granted.
     */
    void deallocate(
        cuda::stream_ref stream,
        void* ptr,
        std::size_t bytes,
        std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT
    ) noexcept {
        auto const amount = safe_cast<std::int64_t>(bytes);
        balance_.fetch_add(amount, std::memory_order_acq_rel);
        adaptor_->total_reserved_.fetch_add(amount, std::memory_order_acq_rel);
        adaptor_->deallocate(stream, ptr, bytes, alignment);
    }

    /**
     * @brief Allocate memory synchronously.
     *
     * Routed through the async path on the adaptor's dedicated sync stream, so the
     * accounting is identical to `allocate()`.
     *
     * @param bytes Number of bytes to allocate.
     * @param alignment Alignment requirement.
     * @return Pointer to the allocated memory.
     *
     * @throws rmm::out_of_memory if @p bytes exceeds the remaining balance.
     */
    void* allocate_sync(
        std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT
    ) {
        auto* ptr = allocate(adaptor_->sync_stream_, bytes, alignment);
        adaptor_->sync_stream_.synchronize();
        return ptr;
    }

    /**
     * @brief Deallocate memory synchronously.
     *
     * @param ptr Pointer to the memory to deallocate.
     * @param bytes Number of bytes to deallocate.
     * @param alignment Alignment of the original allocation.
     */
    void deallocate_sync(
        void* ptr,
        std::size_t bytes,
        std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT
    ) noexcept {
        deallocate(adaptor_->sync_stream_, ptr, bytes, alignment);
        adaptor_->sync_stream_.synchronize_no_throw();
    }

    /**
     * @brief Equality comparison.
     *
     * @param other The other reservation to compare.
     * @return True if the two instances are the same.
     */
    [[nodiscard]] bool operator==(MemoryReservationImpl const& other) const noexcept {
        return this == std::addressof(other);
    }

    /// @brief Tag this resource as device-accessible for the CCCL concept.
    friend void get_property(
        MemoryReservationImpl const&, cuda::mr::device_accessible
    ) noexcept {}

    /**
     * @brief The adaptor that granted the reservation.
     *
     * @return Reference to the adaptor.
     */
    [[nodiscard]] Adaptor const& adaptor() const noexcept {
        return adaptor_;
    }

  private:
    /// @brief Draw @p bytes down from the balance.
    void draw_down_res(std::int64_t bytes) {
        auto balance = balance_.load(std::memory_order_relaxed);
        do {
            RAPIDSMPF_EXPECTS(
                bytes <= balance,
                "allocation of " + format_nbytes(bytes)
                    + " exceeds reservation (grant: " + format_nbytes(grant_)
                    + ", remaining: " + format_nbytes(balance) + ")",
                rmm::out_of_memory
            );
        } while (!balance_.compare_exchange_weak(
            balance, balance - bytes, std::memory_order_acq_rel, std::memory_order_relaxed
        ));
    }

    Adaptor adaptor_;
    std::int64_t const grant_;
    std::size_t const overbooking_;
    std::atomic<std::int64_t> balance_;
};

}  // namespace rapidsmpf::detail
