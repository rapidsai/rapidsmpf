/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstddef>
#include <type_traits>

#include <rmm/error.hpp>
#include <rmm/mr/device_memory_resource.hpp>
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
    using AllocTypeArray = std::array<std::int64_t, 2>;

    /**
     * @brief Returns the total number of allocations performed by the specified allocator
     * type.
     *
     * @param alloc_type The allocator type to query. Use `AllocType::ALL` to return the
     * sum across all types.
     * @return The number of allocations for the specified type.
     */
    [[nodiscard]] std::int64_t num_total_allocs(
        AllocType alloc_type = AllocType::ALL
    ) const noexcept;

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
    [[nodiscard]] std::int64_t num_current_allocs(
        AllocType alloc_type = AllocType::ALL
    ) const noexcept;

    /**
     * @brief Returns the current memory usage in bytes for the specified allocator type.
     *
     * Current usage is the total bytes currently allocated but not yet deallocated.
     *
     * @param alloc_type The allocator type to query. Use `AllocType::ALL` to return the
     * sum across all types.
     * @return The current memory usage in bytes for the specified type.
     */
    [[nodiscard]] std::int64_t current(
        AllocType alloc_type = AllocType::ALL
    ) const noexcept;

    /**
     * @brief Returns the total number of bytes allocated.
     *
     * This value accumulates over time and is not reduced by deallocations.
     *
     * @param alloc_type The allocator type to query. Use `AllocType::ALL` to return
     * the sum across all types.
     * @return The total number of bytes allocated for the specified type.
     */
    [[nodiscard]] std::int64_t total(
        AllocType alloc_type = AllocType::ALL
    ) const noexcept;

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
    [[nodiscard]] std::int64_t peak(AllocType alloc_type = AllocType::ALL) const noexcept;

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
    void record_allocation(AllocType alloc_type, std::int64_t nbytes);

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
    void record_deallocation(AllocType alloc_type, std::int64_t nbytes);

    /**
     * @brief Merge the memory statistics of a subscope into this record.
     *
     * Combines the memory tracking data from a nested scope (subscope) into this
     * record, updating statistics to include the subscope's allocations, peaks,
     * and totals.
     *
     * This method treats the given record as a child scope nested within this scope,
     * so peak usage is updated considering the current usage plus the subscope's peak,
     * reflecting hierarchical (inclusive) memory usage accounting.
     *
     * This design allows memory scopes to be organized hierarchically, so when querying
     * a parent scope, its statistics are **inclusive of all nested scopes** — similar to
     * hierarchical memory profiling tools. However, it assumes that the parent scope's
     * statistics remain constant during the execution of the subscope.
     *
     * @param subscope The scoped memory record representing a completed nested region.
     * @return Reference to this object after merging the subscope.
     *
     * @see add_scope()
     */
    ScopedMemoryRecord& add_subscope(ScopedMemoryRecord const& subscope);

    /**
     * @brief Merge the memory statistics of another scope into this one.
     *
     * Unlike `add_subscope()`, this method treats the given scope as a peer or sibling,
     * rather than a nested child. It aggregates totals and allocation counts and
     * updates peak usage by taking the maximum peaks independently.
     *
     * This is useful for combining memory statistics across multiple independent
     * scopes, such as from different threads or non-nested regions.
     *
     * @param scope The scope to combine with this one.
     * @return Reference to this object after summing.
     *
     * @see add_subscope()
     */
    ScopedMemoryRecord& add_scope(ScopedMemoryRecord const& scope);

  private:
    AllocTypeArray num_current_allocs_{{0, 0}};
    AllocTypeArray num_total_allocs_{{0, 0}};
    AllocTypeArray current_{{0, 0}};
    AllocTypeArray total_{{0, 0}};
    AllocTypeArray peak_{{0, 0}};
    std::int64_t highest_peak_{0};
};

static_assert(
    std::is_trivially_copyable_v<ScopedMemoryRecord>,
    "ScopedMemoryRecord must be trivially copyable"
);

}  // namespace rapidsmpf
