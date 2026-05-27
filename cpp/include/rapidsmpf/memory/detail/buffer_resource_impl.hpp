/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <stack>
#include <thread>
#include <unordered_map>
#include <utility>

#include <cuda/memory_resource>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_pool.hpp>

#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/host_memory_resource.hpp>
#include <rapidsmpf/memory/memory_reservation.hpp>
#include <rapidsmpf/memory/pinned_memory_resource.hpp>
#include <rapidsmpf/memory/resource_types.hpp>
#include <rapidsmpf/memory/scoped_memory_record.hpp>
#include <rapidsmpf/memory/spill_manager.hpp>
#include <rapidsmpf/statistics.hpp>
#include <rapidsmpf/utils/misc.hpp>

namespace rapidsmpf {

class BufferResource;  // defined in buffer_resource.hpp.

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

namespace detail {

/**
 * @brief Implementation class for `BufferResource`.
 *
 * Holds all of `BufferResource`'s state: the user-provided device MR plus
 * per-allocation tracking (lifetime stats + scoped records), pinned/host
 * MRs, reservation bookkeeping, stream pool, spill manager, and statistics.
 *
 * This class satisfies the CCCL `cuda::mr::resource` concept and is held by
 * `BufferResource` via `cuda::mr::shared_resource` for reference-counted
 * ownership.
 */
class BufferResourceImpl {
  public:
    /// @brief Type-erased device-accessible memory resource.
    using any_device_resource = cuda::mr::any_resource<cuda::mr::device_accessible>;

    /**
     * @brief Construct the impl.
     *
     * @param device_mr Primary device memory resource.
     * @param pinned_mr Optional pinned host memory resource. If
     * `PinnedMemoryResource::Disabled`, pinned host allocations fail regardless
     * of `memory_limits`.
     * @param memory_limits Maximum bytes per memory type. Missing entries
     * default to unlimited.
     * @param periodic_spill_check Pause between periodic spill checks
     * (`std::nullopt` disables the dedicated spill thread).
     * @param stream_pool CUDA stream pool used for operations without an
     * explicit stream.
     * @param statistics Statistics instance.
     */
    BufferResourceImpl(
        any_device_resource device_mr,
        std::optional<PinnedMemoryResource> pinned_mr,
        std::unordered_map<MemoryType, std::int64_t> memory_limits,
        std::optional<Duration> periodic_spill_check,
        std::shared_ptr<rmm::cuda_stream_pool> stream_pool,
        std::shared_ptr<Statistics> statistics
    );

    ~BufferResourceImpl() = default;

    BufferResourceImpl(BufferResourceImpl const&) = delete;
    BufferResourceImpl(BufferResourceImpl&&) = delete;
    BufferResourceImpl& operator=(BufferResourceImpl const&) = delete;
    BufferResourceImpl& operator=(BufferResourceImpl&&) = delete;

    /**
     * @brief CCCL concept: async allocation. Records the alloc and forwards
     * to the user-provided device MR.
     *
     * @param stream The CUDA stream for the allocation.
     * @param bytes Number of bytes to allocate.
     * @param alignment Alignment requirement.
     * @return Pointer to the allocated memory.
     */
    void* allocate(
        cuda::stream_ref stream,
        std::size_t bytes,
        std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT
    ) {
        void* ret = device_mr_.allocate(stream, bytes, alignment);
        std::lock_guard<std::mutex> lock(mutex_);
        main_record_.record_allocation(safe_cast<std::int64_t>(bytes));
        if (!record_stacks_.empty()) {
            auto const thread_id = std::this_thread::get_id();
            auto& record = record_stacks_[thread_id];
            if (!record.empty()) {
                record.top().record_allocation(safe_cast<std::int64_t>(bytes));
                RAPIDSMPF_EXPECTS(
                    allocating_threads_.insert({ret, thread_id}).second,
                    "duplicate memory pointer"
                );
            }
        }
        return ret;
    }

    /**
     * @brief CCCL concept: async deallocation. Records the dealloc and
     * forwards to the user-provided device MR.
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
        {
            std::lock_guard<std::mutex> lock(mutex_);
            main_record_.record_deallocation(safe_cast<std::int64_t>(bytes));
            if (!allocating_threads_.empty()) {
                auto const node = allocating_threads_.extract(ptr);
                if (node) {
                    auto thread_id = node.mapped();
                    auto& record = record_stacks_[thread_id];
                    if (!record.empty()) {
                        record.top().record_deallocation(safe_cast<std::int64_t>(bytes));
                    }
                }
            }
        }
        device_mr_.deallocate(stream, ptr, bytes, alignment);
    }

    /**
     * @brief CCCL concept: sync allocation. Allocates on the internal sync
     * stream and synchronizes before returning.
     *
     * @param bytes Number of bytes to allocate.
     * @param alignment Alignment requirement.
     * @return Pointer to the allocated memory.
     */
    void* allocate_sync(
        std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT
    ) {
        auto* ptr = allocate(sync_stream_, bytes, alignment);
        sync_stream_.synchronize();
        return ptr;
    }

    /**
     * @brief CCCL concept: sync deallocation.
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
        deallocate(sync_stream_, ptr, bytes, alignment);
    }

    /**
     * @brief Equality by identity (two impls are equal iff they are the same instance).
     *
     * @param other The other impl to compare.
     * @return True iff @p other is this same instance.
     */
    [[nodiscard]] bool operator==(BufferResourceImpl const& other) const noexcept {
        return this == std::addressof(other);
    }

    /// @brief Tag this resource as device-accessible for the CCCL concept.
    friend void get_property(
        BufferResourceImpl const&, cuda::mr::device_accessible
    ) noexcept {}

    // --- Per-allocation tracking -------------------------------------------

    /// @copydoc rapidsmpf::BufferResource::get_main_record
    [[nodiscard]] ScopedMemoryRecord get_main_record() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return main_record_;
    }

    /// @copydoc rapidsmpf::BufferResource::current_allocated
    [[nodiscard]] std::int64_t current_allocated() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return main_record_.current();
    }

    /// @copydoc rapidsmpf::BufferResource::begin_scoped_memory_record
    void begin_scoped_memory_record() {
        std::lock_guard<std::mutex> lock(mutex_);
        record_stacks_[std::this_thread::get_id()].emplace();
    }

    /// @copydoc rapidsmpf::BufferResource::end_scoped_memory_record
    ScopedMemoryRecord end_scoped_memory_record() {
        std::lock_guard lock(mutex_);
        auto& stack = record_stacks_.at(std::this_thread::get_id());
        RAPIDSMPF_EXPECTS(
            !stack.empty(),
            "calling end_scoped_memory_record() on an empty stack.",
            std::out_of_range
        );
        auto ret = stack.top();
        stack.pop();
        if (!stack.empty()) {
            stack.top().add_subscope(ret);
        }
        return ret;
    }

    // --- BufferResource public API (rich operations) -----------------------

    /// @copydoc rapidsmpf::BufferResource::host_mr
    [[nodiscard]] rmm::host_async_resource_ref host_mr() noexcept;

    /// @copydoc rapidsmpf::BufferResource::pinned_mr
    [[nodiscard]] rmm::host_device_async_resource_ref pinned_mr();

    /// @copydoc rapidsmpf::BufferResource::try_pinned_mr
    [[nodiscard]] std::optional<any_host_device_resource> try_pinned_mr() const noexcept;

    /// @copydoc rapidsmpf::BufferResource::memory_available
    [[nodiscard]] std::int64_t memory_available(MemoryType mem_type) const noexcept;

    /// @copydoc rapidsmpf::BufferResource::set_memory_limit
    void set_memory_limit(MemoryType mem_type, std::int64_t limit) noexcept;

    /// @copydoc rapidsmpf::BufferResource::memory_reserved
    [[nodiscard]] std::size_t memory_reserved(MemoryType mem_type) const {
        return memory_reserved_[static_cast<std::size_t>(mem_type)];
    }

    /**
     * @copydoc rapidsmpf::BufferResource::reserve
     *
     * @param outer_br Outer `BufferResource` handle stored in the returned
     * `MemoryReservation` so that `MemoryReservation::br()` keeps working
     * with the public `BufferResource` API. The caller must ensure the outer
     * handle outlives the returned reservation.
     */
    std::pair<MemoryReservation, std::size_t> reserve(
        BufferResource* outer_br,
        MemoryType mem_type,
        std::size_t size,
        AllowOverbooking allow_overbooking
    );

    /**
     * @copydoc rapidsmpf::BufferResource::reserve_device_memory_and_spill
     *
     * @param outer_br Outer `BufferResource` handle stored in the returned
     * `MemoryReservation`. The caller must ensure the outer handle outlives
     * the returned reservation.
     */
    MemoryReservation reserve_device_memory_and_spill(
        BufferResource* outer_br, std::size_t size, AllowOverbooking allow_overbooking
    );

    /// @copydoc rapidsmpf::BufferResource::release
    std::size_t release(MemoryReservation& reservation, std::size_t size);

    // The buffer-creating methods need the outer `BufferResource` handle to
    // construct `rmm::device_buffer` instances against an
    // `rmm::device_async_resource_ref` (which requires a copyable resource —
    // the impl itself is non-copyable; the outer handle's shared-ownership
    // state satisfies the requirement).

    // clang-format off
    /**
     * @copydoc rapidsmpf::BufferResource::make_buffer(std::size_t,rmm::cuda_stream_view,MemoryReservation&)
     *
     * @param outer_br Outer `BufferResource` handle used to construct an
     * `rmm::device_async_resource_ref` for device allocations (the impl
     * itself is non-copyable; the outer handle's shared-ownership state
     * satisfies the ref's copyable requirement).
     */
    std::unique_ptr<Buffer> make_buffer(
        BufferResource* outer_br,
        std::size_t size,
        rmm::cuda_stream_view stream,
        MemoryReservation& reservation
    );

    /**
     * @copydoc rapidsmpf::BufferResource::make_buffer(rmm::cuda_stream_view,MemoryReservation&&)
     *
     * @param outer_br Outer `BufferResource` handle. See the other
     * `make_buffer` overload for the rationale.
     */
    std::unique_ptr<Buffer> make_buffer(
        BufferResource* outer_br,
        rmm::cuda_stream_view stream,
        MemoryReservation&& reservation
    );

    /** @copydoc rapidsmpf::BufferResource::move(std::unique_ptr<rmm::device_buffer>,rmm::cuda_stream_view) */
    std::unique_ptr<Buffer> move(
        std::unique_ptr<rmm::device_buffer> data, rmm::cuda_stream_view stream
    );

    /**
     * @copydoc rapidsmpf::BufferResource::move(std::unique_ptr<Buffer>,MemoryReservation&)
     *
     * @param outer_br Outer `BufferResource` handle. Forwarded to
     * `make_buffer` when a copy across memory types is needed.
     */
    std::unique_ptr<Buffer> move(
        BufferResource* outer_br,
        std::unique_ptr<Buffer> buffer,
        MemoryReservation& reservation
    );
    // clang-format on

    /**
     * @copydoc rapidsmpf::BufferResource::move_to_device_buffer
     *
     * @param outer_br Outer `BufferResource` handle. Forwarded to
     * `make_buffer` when a copy to device memory is needed.
     */
    std::unique_ptr<rmm::device_buffer> move_to_device_buffer(
        BufferResource* outer_br,
        std::unique_ptr<Buffer> buffer,
        MemoryReservation& reservation
    );

    /**
     * @copydoc rapidsmpf::BufferResource::move_to_host_buffer
     *
     * @param outer_br Outer `BufferResource` handle. Forwarded to
     * `make_buffer` when a copy to host memory is needed.
     */
    std::unique_ptr<HostBuffer> move_to_host_buffer(
        BufferResource* outer_br,
        std::unique_ptr<Buffer> buffer,
        MemoryReservation& reservation
    );

    /// @copydoc rapidsmpf::BufferResource::stream_pool
    [[nodiscard]] rmm::cuda_stream_pool const& stream_pool() const {
        return *stream_pool_;
    }

    /// @copydoc rapidsmpf::BufferResource::spill_manager
    SpillManager& spill_manager() {
        return spill_manager_;
    }

    /// @copydoc rapidsmpf::BufferResource::statistics
    [[nodiscard]] std::shared_ptr<Statistics> statistics() const noexcept {
        return statistics_;
    }

  private:
    /// @brief Protects all tracking + reservation state.
    mutable std::mutex mutex_;

    /// @brief User-provided primary device MR.
    any_device_resource device_mr_;
    /// @brief Lifetime-of-resource allocation stats.
    ScopedMemoryRecord main_record_;
    /// @brief Per-thread scoped record stacks.
    std::unordered_map<std::thread::id, std::stack<ScopedMemoryRecord>> record_stacks_;
    /// @brief Map from allocation ptr → originating thread, used to credit
    /// deallocations back to the right per-thread scoped record.
    std::unordered_map<void*, std::thread::id> allocating_threads_;
    /// @brief Stream used for synchronous allocations/deallocations.
    rmm::cuda_stream sync_stream_{rmm::cuda_stream::flags::non_blocking};

    std::optional<PinnedMemoryResource> pinned_mr_;
    HostMemoryResource host_mr_;
    std::array<std::atomic<std::int64_t>, MEMORY_TYPES.size()> memory_limits_;
    /// @brief Zero-initialized reserved counters per memory type.
    std::array<std::size_t, MEMORY_TYPES.size()> memory_reserved_ = {};
    std::shared_ptr<rmm::cuda_stream_pool> stream_pool_;
    SpillManager spill_manager_;
    std::shared_ptr<Statistics> statistics_;
};

}  // namespace detail

}  // namespace rapidsmpf
