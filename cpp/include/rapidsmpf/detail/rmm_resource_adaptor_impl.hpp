/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <stack>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <cuda_runtime_api.h>

#include <cuda/memory_resource>

#include <rmm/aligned.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/error.hpp>
#include <rmm/resource_ref.hpp>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/scoped_memory_record.hpp>
#include <rapidsmpf/utils/misc.hpp>

namespace rapidsmpf::detail {

/**
 * @brief Implementation class for RmmResourceAdaptor.
 *
 * Holds all mutable state for memory tracking and fallback allocation.
 * This class satisfies the CCCL `cuda::mr::resource` concept and is held by
 * `RmmResourceAdaptor` via `cuda::mr::shared_resource` for
 * reference-counted ownership.
 *
 * @tparam PrimaryMR The type of the primary memory resource. Use a concrete
 * resource type (e.g. `cuda::pinned_memory_pool`) to store the resource
 * directly inside the shared control block, avoiding an extra heap allocation.
 * @tparam FallbackMR The type of the optional fallback memory resource, used
 * when the primary resource throws `rmm::out_of_memory`. Defaults to
 * `cuda::mr::any_resource<cuda::mr::device_accessible>` for type-erased use.
 */
template <
    cuda::mr::resource_with<cuda::mr::device_accessible> PrimaryMR,
    cuda::mr::resource_with<cuda::mr::device_accessible> FallbackMR =
        cuda::mr::any_resource<cuda::mr::device_accessible>>
class RmmResourceAdaptorImpl {
  public:
    /**
     * @brief Construct with a primary and optional fallback memory resource.
     *
     * @param primary_mr The primary memory resource (moved in).
     * @param fallback_mr Optional fallback memory resource.
     */
    // NOLINTBEGIN(clang-analyzer-core.StackAddressEscape): false positive — primary_mr
    // and fallback_mr are moved into a heap-allocated control block inside
    // make_shared_resource; the analyzer incorrectly traces the forwarding
    // reference chain back to the outer caller's stack frame.
    RmmResourceAdaptorImpl(  // NOLINT(clang-analyzer-core.StackAddressEscape)
        PrimaryMR primary_mr,
        std::optional<FallbackMR> fallback_mr = std::nullopt
    )
        : primary_mr_{std::move(primary_mr)}, fallback_mr_{std::move(fallback_mr)} {}

    // NOLINTEND(clang-analyzer-core.StackAddressEscape)

    /**
     * @brief Construct the primary resource in-place from forwarded arguments.
     *
     * Use this overload when the primary resource type is non-movable (e.g.
     * `cuda::pinned_memory_pool`). The resource is constructed directly inside
     * the shared-ownership control block, avoiding an extra heap allocation.
     *
     * @param args Arguments forwarded to the `PrimaryMR` constructor.
     */
    template <typename... Args>
    explicit RmmResourceAdaptorImpl(std::in_place_t, Args&&... args)
        : primary_mr_{std::forward<Args>(args)...}, fallback_mr_{std::nullopt} {}

    ~RmmResourceAdaptorImpl() = default;

    RmmResourceAdaptorImpl(RmmResourceAdaptorImpl const&) = delete;
    RmmResourceAdaptorImpl(RmmResourceAdaptorImpl&&) = delete;
    RmmResourceAdaptorImpl& operator=(RmmResourceAdaptorImpl const&) = delete;
    RmmResourceAdaptorImpl& operator=(RmmResourceAdaptorImpl&&) = delete;

    /**
     * @brief Equality comparison.
     *
     * @param other The other impl to compare.
     * @return True if the two instances are the same.
     */
    [[nodiscard]] bool operator==(RmmResourceAdaptorImpl const& other) const noexcept {
        return this == std::addressof(other);
    }

    /**
     * @brief Returns a reference to the primary upstream resource.
     * @return Reference to the primary resource.
     */
    [[nodiscard]] PrimaryMR const& get_upstream_resource() const noexcept {
        return primary_mr_;
    }

    /**
     * @brief Returns a const reference to the optional fallback resource.
     * @return Const reference to the optional fallback resource, or `std::nullopt` if
     * no fallback is configured.
     */
    [[nodiscard]] std::optional<FallbackMR> const&
    get_fallback_resource() const noexcept {
        return fallback_mr_;
    }

    /// @copydoc RmmResourceAdaptor::get_main_record
    [[nodiscard]] ScopedMemoryRecord get_main_record() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return main_record_;
    }

    /// @copydoc RmmResourceAdaptor::current_allocated
    [[nodiscard]] std::int64_t current_allocated() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return main_record_.current();
    }

    /// @copydoc RmmResourceAdaptor::begin_scoped_memory_record
    void begin_scoped_memory_record() {
        std::lock_guard<std::mutex> lock(mutex_);
        record_stacks_[std::this_thread::get_id()].emplace();
    }

    /// @copydoc RmmResourceAdaptor::end_scoped_memory_record
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

    /**
     * @brief Allocate memory asynchronously on the given stream.
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
        constexpr auto PRIMARY = ScopedMemoryRecord::AllocType::PRIMARY;
        constexpr auto FALLBACK = ScopedMemoryRecord::AllocType::FALLBACK;

        void* ret{};
        auto alloc_type = PRIMARY;
        try {
            ret = primary_mr_.allocate(stream, bytes, alignment);
        } catch (rmm::out_of_memory const& e) {
            if (fallback_mr_.has_value()) {
                alloc_type = FALLBACK;
                ret = fallback_mr_->allocate(stream, bytes, alignment);
                std::lock_guard<std::mutex> lock(mutex_);
                fallback_allocations_.insert(ret);
            } else {
                throw;
            }
        }
        std::lock_guard<std::mutex> lock(mutex_);
        main_record_.record_allocation(alloc_type, safe_cast<std::int64_t>(bytes));
        if (!record_stacks_.empty()) {
            auto const thread_id = std::this_thread::get_id();
            auto& record = record_stacks_[thread_id];
            if (!record.empty()) {
                record.top().record_allocation(
                    alloc_type, safe_cast<std::int64_t>(bytes)
                );
                RAPIDSMPF_EXPECTS(
                    allocating_threads_.insert({ret, thread_id}).second,
                    "duplicate memory pointer"
                );
            }
        }
        return ret;
    }

    /**
     * @brief Deallocate memory asynchronously on the given stream.
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
        constexpr auto PRIMARY = ScopedMemoryRecord::AllocType::PRIMARY;
        constexpr auto FALLBACK = ScopedMemoryRecord::AllocType::FALLBACK;

        ScopedMemoryRecord::AllocType alloc_type;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            alloc_type = (fallback_allocations_.erase(ptr) == 0) ? PRIMARY : FALLBACK;
            main_record_.record_deallocation(alloc_type, safe_cast<std::int64_t>(bytes));
            if (!allocating_threads_.empty()) {
                auto const node = allocating_threads_.extract(ptr);
                if (node) {
                    auto thread_id = node.mapped();
                    auto& record = record_stacks_[thread_id];
                    if (!record.empty()) {
                        record.top().record_deallocation(
                            alloc_type, safe_cast<std::int64_t>(bytes)
                        );
                    }
                }
            }
        }
        if (alloc_type == PRIMARY) {
            primary_mr_.deallocate(stream, ptr, bytes, alignment);
        } else {
            fallback_mr_->deallocate(stream, ptr, bytes, alignment);
        }
    }

    /**
     * @brief Allocate memory synchronously.
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
        deallocate(sync_stream_, ptr, bytes, alignment);
    }

    /// @brief Tag this resource as device-accessible for the CCCL concept.
    friend void get_property(
        RmmResourceAdaptorImpl const&, cuda::mr::device_accessible
    ) noexcept {}

  private:
    mutable std::mutex mutex_;
    PrimaryMR primary_mr_;
    std::optional<FallbackMR> fallback_mr_;
    std::unordered_set<void*> fallback_allocations_;

    ScopedMemoryRecord main_record_;
    std::unordered_map<std::thread::id, std::stack<ScopedMemoryRecord>> record_stacks_;
    std::unordered_map<void*, std::thread::id> allocating_threads_;

    rmm::cuda_stream sync_stream_{
        rmm::cuda_stream::flags::non_blocking
    };  ///< Stream for synchronous allocations and deallocations.
};

}  // namespace rapidsmpf::detail
