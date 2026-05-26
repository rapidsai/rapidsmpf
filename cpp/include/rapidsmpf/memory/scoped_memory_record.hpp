/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <type_traits>

namespace rapidsmpf {

/**
 * @brief Memory statistics for a specific scope.
 *
 * @note Is trivially copyable.
 */
struct ScopedMemoryRecord {
    /**
     * @brief Returns the total number of allocations performed.
     *
     * @return The total number of allocations.
     */
    [[nodiscard]] std::int64_t num_total_allocs() const noexcept;

    /**
     * @brief Returns the number of currently active (non-deallocated) allocations.
     *
     * @return The number of active allocations.
     */
    [[nodiscard]] std::int64_t num_current_allocs() const noexcept;

    /**
     * @brief Returns the current memory usage in bytes.
     *
     * Current usage is the total bytes currently allocated but not yet deallocated.
     *
     * @return The current memory usage in bytes.
     */
    [[nodiscard]] std::int64_t current() const noexcept;

    /**
     * @brief Returns the total number of bytes allocated.
     *
     * This value accumulates over time and is not reduced by deallocations.
     *
     * @return The total number of bytes allocated.
     */
    [[nodiscard]] std::int64_t total() const noexcept;

    /**
     * @brief Returns the peak memory usage (in bytes).
     *
     * The peak represents the highest value reached by current memory usage over the
     * lifetime of the scope, including contributions from merged subscopes.
     *
     * @return The peak memory usage in bytes.
     */
    [[nodiscard]] std::int64_t peak() const noexcept;

    /**
     * @brief Returns the size of the largest single allocation (in bytes).
     *
     * @return The largest single allocation in bytes.
     */
    [[nodiscard]] std::int64_t max() const noexcept;

    /**
     * @brief Records a memory allocation event.
     *
     * Updates the allocation counters and memory usage statistics, and adjusts
     * peak usage if the new current usage exceeds the previous peak.
     *
     * @param nbytes The number of bytes allocated.
     *
     * @note Is not thread-safe.
     */
    void record_allocation(std::int64_t nbytes);

    /**
     * @brief Records a memory deallocation event.
     *
     * Reduces the current memory usage.
     *
     * @param nbytes The number of bytes deallocated.
     *
     * @note Is not thread-safe.
     */
    void record_deallocation(std::int64_t nbytes);

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
    std::int64_t num_current_allocs_{0};
    std::int64_t num_total_allocs_{0};
    std::int64_t current_{0};
    std::int64_t total_{0};
    std::int64_t peak_{0};
    std::int64_t max_{0};
};

static_assert(
    std::is_trivially_copyable_v<ScopedMemoryRecord>,
    "ScopedMemoryRecord must be trivially copyable"
);

}  // namespace rapidsmpf
