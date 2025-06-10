/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstddef>
#include <mutex>
#include <optional>
#include <stack>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

#include <rmm/error.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

namespace rapidsmpf {

/**
 * @brief Memory statistics for a specific scope.
 *
 * @note Is trivially copyable.
 */
struct ScopedMemoryRecord {
    /// Allocation source types.
    enum class AllocType : std::size_t {
        PRIMARY = 0,  ///< The primary allocator (first-choice allocator).
        FALLBACK = 1,  ///< The fallback allocator (used when the primary fails).
        ALL = 2  ///< Aggregated statistics from both primary and fallback allocators.
    };

    /// Array type for storing per-allocator statistics.
    using AllocTypeArray = std::array<std::uint64_t, 2>;

    /**
     * @brief Returns the total number of allocations performed by the specified allocator
     * type.
     *
     * @param alloc_type The allocator type to query. Use `AllocType::ALL` to return the
     * sum across all types.
     * @return The number of allocations for the specified type.
     */
    [[nodiscard]] std::uint64_t num_total_allocs(AllocType alloc_type = AllocType::ALL)
        const noexcept;

    /**
     * @brief Returns the number of currently active (non-deallocated) allocations
     *        for the specified allocator type.
     *
     * This reflects the number of allocations that have not yet been deallocated.
     *
     * @param alloc_type The allocator type to query. Use `AllocType::ALL` to return the
     * sum across all types.
     * @return The number of active allocations for the specified type.
     */
    [[nodiscard]] std::uint64_t num_current_allocs(AllocType alloc_type = AllocType::ALL)
        const noexcept;

    /**
     * @brief Returns the current memory usage in bytes for the specified allocator type.
     *
     * Current usage is the total bytes currently allocated but not yet deallocated.
     *
     * @param alloc_type The allocator type to query. Use `AllocType::ALL` to return the
     * sum across all types.
     * @return The current memory usage in bytes for the specified type.
     */
    [[nodiscard]] std::uint64_t current(AllocType alloc_type = AllocType::ALL)
        const noexcept;

    /**
     * @brief Returns the total number of bytes allocated over the lifetime of this
     * main_record.
     *
     * This value accumulates over time and is not reduced by deallocations.
     *
     * @param alloc_type The allocator type to query. Use `AllocType::ALL` to return
     * the sum across all types.
     * @return The total number of bytes allocated for the specified type.
     */
    [[nodiscard]] std::uint64_t total(AllocType alloc_type = AllocType::ALL)
        const noexcept;

    /**
     * @brief Returns the peak memory usage (in bytes) for the specified allocator type.
     *
     * The peak represents the highest value reached by current memory usage over the
     * lifetime of the allocator. It does not decrease after deallocations.
     *
     * When queried with `AllocType::ALL`, this returns the highest combined memory
     * usage ever observed across both primary and fallback allocators, not the sum of
     * individual peaks.
     *
     * @param alloc_type The allocator type to query. Defaults to `AllocType::ALL`.
     * @return The peak memory usage in bytes for the specified type.
     */
    [[nodiscard]] std::uint64_t peak(AllocType alloc_type = AllocType::ALL)
        const noexcept;

    /**
     * @brief Records a memory allocation event.
     *
     * Updates the allocation counters and memory usage statistics, and adjusts
     * peak usage if the new current usage exceeds the previous peak.
     *
     * @param alloc_type The allocator that performed the allocation.
     * @param nbytes     The number of bytes allocated.
     *
     * @note Is not thread-safe.
     */
    void record_allocation(AllocType alloc_type, std::uint64_t nbytes);

    /**
     * @brief Records a memory deallocation event.
     *
     * Reduces the current memory usage for the specified allocator.
     *
     * @param alloc_type The allocator that performed the deallocation.
     * @param nbytes     The number of bytes deallocated.
     *
     * @note Is not thread-safe.
     */
    void record_deallocation(AllocType alloc_type, std::uint64_t nbytes);

  private:
    AllocTypeArray num_current_allocs_{{0, 0}};
    AllocTypeArray num_total_allocs_{{0, 0}};
    AllocTypeArray current_{{0, 0}};
    AllocTypeArray total_{{0, 0}};
    AllocTypeArray peak_{{0, 0}};
    std::uint64_t highest_peak_{0};
};

static_assert(
    std::is_trivially_copyable<ScopedMemoryRecord>::value,
    "ScopedMemoryRecord must be trivially copyable"
);

/**
 * @brief A RMM memory resource adaptor tailored to RapidsMPF.
 *
 * This adaptor implements:
 * - Memory usage tracking.
 * - Fallback memory resource support upon out-of-memory in the primary resource.
 */
class RmmResourceAdaptor final : public rmm::mr::device_memory_resource {
  public:
    /**
     * @brief Construct with specified primary and optional fallback memory resource.
     *
     * @param primary_mr The primary memory resource.
     * @param fallback_mr Optional fallback memory resource.
     */
    RmmResourceAdaptor(
        rmm::device_async_resource_ref primary_mr,
        std::optional<rmm::device_async_resource_ref> fallback_mr = std::nullopt
    )
        : primary_mr_{primary_mr}, fallback_mr_{fallback_mr} {}

    RmmResourceAdaptor() = delete;
    ~RmmResourceAdaptor() override = default;

    /**
     * @brief Get a reference to the primary upstream resource.
     *
     * @return Reference to the RMM memory resource.
     */
    [[nodiscard]] rmm::device_async_resource_ref get_upstream_resource() const noexcept {
        return primary_mr_;
    }

    /**
     * @brief Get a reference to the fallback upstream resource.
     *
     * This resource is used if the primary resource throws `rmm::out_of_memory`.
     *
     * @return Optional reference to the fallback RMM memory resource.
     */
    [[nodiscard]] std::optional<rmm::device_async_resource_ref> get_fallback_resource(
    ) const noexcept {
        return fallback_mr_;
    }

    /**
     * @brief Get a copy of the tracked main_record.
     *
     * @return Scoped memory main_record instance.
     */
    [[nodiscard]] ScopedMemoryRecord get_main_record() const;

    /**
     * @brief Get the total current allocated memory from both primary and fallback.
     *
     * @return Total number of currently allocated bytes.
     */
    [[nodiscard]] std::uint64_t current_allocated() const noexcept;

  private:
    /**
     * @brief Allocates memory of size at least `bytes` using the upstream resource.
     *
     * Attempts to allocate using the primary resource. If it fails with
     * `rmm::out_of_memory` and a fallback is provided, retries allocation using the
     * fallback.
     *
     * @param bytes Number of bytes to allocate.
     * @param stream CUDA stream to associate with this allocation.
     * @return Pointer to the allocated memory.
     *
     * @throws rmm::out_of_memory or other exceptions from the upstream resources.
     */
    void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) override;

    /**
     * @brief Deallocates memory previously allocated via this resource.
     *
     * @param ptr Pointer to the memory to deallocate.
     * @param bytes Size of the allocation in bytes.
     * @param stream CUDA stream to associate with this deallocation.
     */
    void do_deallocate(void* ptr, std::size_t bytes, rmm::cuda_stream_view stream)
        override;

    /**
     * @brief Check if this memory resource is equal to another.
     *
     * Equality is defined by comparing the primary and fallback resources.
     *
     * @param other Another memory resource to compare against.
     * @return true if both resources are equivalent, false otherwise.
     */
    [[nodiscard]] bool do_is_equal(rmm::mr::device_memory_resource const& other
    ) const noexcept override;

    mutable std::mutex mutex_;
    rmm::device_async_resource_ref primary_mr_;
    std::optional<rmm::device_async_resource_ref> fallback_mr_;
    std::unordered_set<void*> fallback_allocations_;
    ScopedMemoryRecord main_record_;
    std::unordered_map<std::thread::id, std::stack<ScopedMemoryRecord>> record_stacks_;
    std::unordered_map<void*, std::thread::id> allocating_threads_;
};


}  // namespace rapidsmpf
