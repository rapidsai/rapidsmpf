/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>

#include <cuda/memory_resource>

#include <rmm/aligned.hpp>
#include <rmm/error.hpp>

#include <rapidsmpf/detail/rmm_resource_adaptor_impl.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/utils/misc.hpp>
#include <rapidsmpf/utils/string.hpp>

namespace rapidsmpf::detail {

template <cuda::mr::resource_with<cuda::mr::device_accessible> PrimaryMR>
class ReservationImpl;

/**
 * @brief Implementation class for ReservationAwareResourceAdaptor.
 *
 * Extends `RmmResourceAdaptorImpl` with a memory limit and a running total of the
 * memory currently held by live reservations. Availability is
 * `limit - allocated - total_reserved`, where `allocated` is the tracking counter
 * inherited from the base.
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

    ReservationAwareResourceAdaptorImpl(ReservationAwareResourceAdaptorImpl const&) =
        delete;
    ReservationAwareResourceAdaptorImpl(ReservationAwareResourceAdaptorImpl&&) = delete;
    ReservationAwareResourceAdaptorImpl& operator=(
        ReservationAwareResourceAdaptorImpl const&
    ) = delete;
    ReservationAwareResourceAdaptorImpl& operator=(
        ReservationAwareResourceAdaptorImpl&&
    ) = delete;

    /// @copydoc ReservationAwareResourceAdaptor::limit
    [[nodiscard]] std::int64_t limit() const noexcept {
        return limit_.load(std::memory_order_acquire);
    }

    /// @copydoc ReservationAwareResourceAdaptor::set_limit
    void set_limit(std::int64_t limit) noexcept {
        limit_.store(limit, std::memory_order_release);
    }

    /// @copydoc ReservationAwareResourceAdaptor::total_reserved
    [[nodiscard]] std::int64_t total_reserved() const noexcept {
        return total_reserved_.load(std::memory_order_acquire);
    }

    /// @copydoc ReservationAwareResourceAdaptor::available
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
    [[nodiscard]] std::pair<std::int64_t, std::size_t> try_reserve(
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
                return {want, overbooking};
            }
        }
    }

  private:
    friend class ReservationImpl<PrimaryMR>;

    /// @brief Bytes stop being reserved, either by being allocated or by being
    /// refunded when their owner releases.
    void decrement_reserved(std::int64_t nbytes) noexcept {
        total_reserved_.fetch_sub(nbytes, std::memory_order_acq_rel);
    }

    /// @brief Freed bytes go back to the reservation that allocated them.
    void increment_reserved(std::int64_t nbytes) noexcept {
        total_reserved_.fetch_add(nbytes, std::memory_order_acq_rel);
    }

    std::atomic<std::int64_t> limit_;
    std::atomic<std::int64_t> total_reserved_{0};
};

/**
 * @brief Shared state of a memory reservation.
 *
 * Satisfies the `cuda::mr::resource` concept, so this is what ends up stored inside
 * `rmm::device_buffer`'s `cuda::mr::any_resource` member for every buffer allocated
 * from the reservation. It is held by `cuda::mr::shared_resource`, meaning copies
 * share this state and keep it alive for as long as any derived buffer exists.
 *
 * @par Owner lifetime
 *
 * Exactly one holder is the *owner* (`MemoryReservation2`); every other reference is a
 * copy made by RMM when a buffer stored the resource. The owner calls
 * `release_owner()` when its scope ends, which refunds the unused balance
 * immediately rather than waiting for the last derived buffer to die. Buffer-held
 * copies never call it, so the refund is tied to the reserving scope rather than to
 * allocation lifetime.
 *
 * Before release, the reservation is a hard cap: an allocation exceeding the
 * remaining balance throws. After release, allocations fall through to the adaptor
 * and are tracked but unreserved, which keeps `rmm::device_buffer::resize()` and
 * `memory_resource()` propagation working on buffers that outlive their reservation.
 *
 * @tparam PrimaryMR The type of the primary memory resource.
 */
template <cuda::mr::resource_with<cuda::mr::device_accessible> PrimaryMR>
class ReservationImpl {
  public:
    using Adaptor = ReservationAwareResourceAdaptorImpl<PrimaryMR>;  ///< Adaptor type.

    /**
     * @brief Construct a reservation over an already-granted number of bytes.
     *
     * @param adaptor The adaptor that granted the reservation.
     * @param grant The number of bytes granted.
     */
    ReservationImpl(cuda::mr::shared_resource<Adaptor> adaptor, std::int64_t grant)
        : adaptor_{std::move(adaptor)}, grant_{grant}, balance_{grant} {}

    ~ReservationImpl() = default;

    ReservationImpl(ReservationImpl const&) = delete;
    ReservationImpl(ReservationImpl&&) = delete;
    ReservationImpl& operator=(ReservationImpl const&) = delete;
    ReservationImpl& operator=(ReservationImpl&&) = delete;

    /**
     * @brief Relinquish the unused balance back to the adaptor.
     *
     * Called by the owning `MemoryReservation2`; idempotent. Allocations already made
     * stay valid and this object stays alive as long as any buffer references it, but
     * it no longer draws on a reservation.
     */
    void release_owner() noexcept {
        std::int64_t remaining = 0;
        {
            std::lock_guard const lock(mutex_);
            if (released_) {
                return;
            }
            remaining = std::exchange(balance_, 0);
            released_ = true;
        }
        // Outside the lock, so the reservation lock and the adaptor's counters are
        // never held at the same time.
        adaptor_->decrement_reserved(remaining);
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
     * @brief The remaining balance.
     *
     * @return The number of unallocated bytes, or zero once the owner has released.
     */
    [[nodiscard]] std::int64_t balance() const noexcept {
        std::lock_guard const lock(mutex_);
        return balance_;
    }

    /**
     * @brief Whether the owning `MemoryReservation2` has released this reservation.
     *
     * @return True if the owner has released.
     */
    [[nodiscard]] bool released() const noexcept {
        std::lock_guard const lock(mutex_);
        return released_;
    }

    /**
     * @brief Allocate memory asynchronously on the given stream.
     *
     * The balance is drawn down before allocating, which is what enforces the cap, while
     * the adaptor's reserved counter is only decremented once the base has recorded the
     * allocation, so a concurrent `available()` never over-reports.
     *
     * @param stream The CUDA stream for the allocation.
     * @param bytes Number of bytes to allocate.
     * @param alignment Alignment requirement.
     * @return Pointer to the allocated memory.
     *
     * @throws rmm::out_of_memory if the reservation is live and @p bytes exceeds the
     * remaining balance.
     */
    void* allocate(
        cuda::stream_ref stream,
        std::size_t bytes,
        std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT
    ) {
        auto const padded_bytes =
            safe_cast<std::int64_t>(rmm::align_up(bytes, alignment));
        // Zero once the owner has released, which makes the calls below no-ops.
        std::int64_t const drawn = draw_down_res(padded_bytes);
        void* ptr = nullptr;
        try {
            ptr = adaptor_->allocate(stream, bytes, alignment);
        } catch (...) {
            add_back_res(drawn);  // The allocation never happened.
            throw;
        }
        adaptor_->decrement_reserved(drawn);
        return ptr;
    }

    /**
     * @brief Deallocate memory asynchronously on the given stream.
     *
     * Mirrors `allocate()`: the adaptor's reserved counter is restored before the base
     * drops the allocation, so a concurrent `available()` never over-reports.
     *
     * @param stream The CUDA stream for the deallocation.
     * @param ptr Pointer to the memory to deallocate.
     * @param bytes Number of bytes to deallocate.
     * @param alignment Alignment of the original allocation.
     */
    void deallocate(
        cuda::stream_ref stream,
        void* ptr,
        std::size_t bytes,
        std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT
    ) noexcept {
        auto const padded_bytes =
            safe_cast<std::int64_t>(rmm::align_up(bytes, alignment));
        adaptor_->increment_reserved(add_back_res(padded_bytes));
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
     * @throws rmm::out_of_memory if the reservation is live and @p bytes exceeds the
     * remaining balance.
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
    }

    /**
     * @brief Equality comparison.
     *
     * @param other The other reservation to compare.
     * @return True if the two instances are the same.
     */
    [[nodiscard]] bool operator==(ReservationImpl const& other) const noexcept {
        return this == std::addressof(other);
    }

    /// @brief Tag this resource as device-accessible for the CCCL concept.
    friend void get_property(
        ReservationImpl const&, cuda::mr::device_accessible
    ) noexcept {}

  private:
    /**
     * @brief Draw @p padded_bytes down from the balance.
     *
     * @param padded_bytes The number of bytes to draw down.
     * @return The number of bytes drawn: @p padded_bytes, or zero once the owner has
     * released and the allocation is unreserved (but still tracked).
     *
     * @throws rmm::out_of_memory if the owner is alive and the balance is insufficient.
     */
    std::int64_t draw_down_res(std::int64_t padded_bytes) {
        std::lock_guard const lock(mutex_);
        if (released_) {
            return 0;
        }
        RAPIDSMPF_EXPECTS(
            padded_bytes <= balance_,
            "allocation of " + format_nbytes(padded_bytes)
                + " exceeds reservation (grant: " + format_nbytes(grant_)
                + ", remaining: " + format_nbytes(balance_) + ")",
            rmm::out_of_memory
        );
        balance_ -= padded_bytes;
        return padded_bytes;
    }

    /**
     * @brief Add @p padded_bytes back to the balance, capped at the original grant.
     *
     * The cap bounds the inflation caused by freeing, through this reservation, memory
     * that was allocated elsewhere; a reservation that only frees what it allocated
     * never reaches it. Exceeding the grant cannot be reported as an error instead,
     * because this runs on the deallocation path: `cuda::mr::shared_resource` forwards
     * it through a `noexcept` function and `rmm::device_buffer` calls it from its
     * destructor.
     *
     * @param padded_bytes The number of bytes to add back.
     * @return The number of bytes actually added back: less than @p padded_bytes when
     * the cap bites, and zero once the owner has released.
     */
    std::int64_t add_back_res(std::int64_t padded_bytes) noexcept {
        std::lock_guard const lock(mutex_);
        if (released_) {
            return 0;
        }
        auto const added_back = std::min(grant_ - balance_, padded_bytes);
        balance_ += added_back;
        return added_back;
    }

    mutable std::mutex mutex_;
    cuda::mr::shared_resource<Adaptor> adaptor_;
    std::int64_t const grant_;
    std::int64_t balance_;
    bool released_{false};
};

}  // namespace rapidsmpf::detail
