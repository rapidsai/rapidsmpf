/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstddef>
#include <mutex>
#include <numeric>
#include <optional>
#include <stack>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

#include <rmm/error.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <rapidsmpf/error.hpp>

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
    [[nodiscard]] constexpr std::int64_t num_total_allocs(
        AllocType alloc_type = AllocType::ALL
    ) const noexcept {
        return get_or_accumulate(num_total_allocs_, alloc_type);
    }

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
    [[nodiscard]] constexpr std::int64_t num_current_allocs(
        AllocType alloc_type = AllocType::ALL
    ) const noexcept {
        return get_or_accumulate(num_current_allocs_, alloc_type);
    }

    /**
     * @brief Returns the current memory usage in bytes for the specified allocator type.
     *
     * Current usage is the total bytes currently allocated but not yet deallocated.
     *
     * @param alloc_type The allocator type to query. Use `AllocType::ALL` to return the
     * sum across all types.
     * @return The current memory usage in bytes for the specified type.
     */
    [[nodiscard]] constexpr std::int64_t current(AllocType alloc_type = AllocType::ALL)
        const noexcept {
        return get_or_accumulate(current_, alloc_type);
    }

    /**
     * @brief Returns the total number of bytes allocated.
     *
     * This value accumulates over time and is not reduced by deallocations.
     *
     * @param alloc_type The allocator type to query. Use `AllocType::ALL` to return
     * the sum across all types.
     * @return The total number of bytes allocated for the specified type.
     */
    [[nodiscard]] constexpr std::int64_t total(AllocType alloc_type = AllocType::ALL)
        const noexcept {
        return get_or_accumulate(total_, alloc_type);
    }

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
    [[nodiscard]] constexpr std::int64_t peak(AllocType alloc_type = AllocType::ALL)
        const noexcept {
        if (alloc_type == AllocType::ALL) {
            return highest_peak_;
        }
        return peak_[static_cast<std::size_t>(alloc_type)];
    }

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
    constexpr void record_allocation(AllocType alloc_type, std::int64_t nbytes) {
        RAPIDSMPF_EXPECTS(
            alloc_type != AllocType::ALL,
            "AllocType::ALL may not be used to record allocation"
        );
        auto at = static_cast<std::size_t>(alloc_type);
        ++num_total_allocs_[at];
        ++num_current_allocs_[at];
        current_[at] += nbytes;
        total_[at] += nbytes;
        peak_[at] = std::max(peak_[at], current_[at]);
        highest_peak_ = std::max(highest_peak_, current());
    }

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
    constexpr void record_deallocation(AllocType alloc_type, std::int64_t nbytes) {
        RAPIDSMPF_EXPECTS(
            alloc_type != AllocType::ALL,
            "AllocType::ALL may not be used to record deallocation"
        );
        auto at = static_cast<std::size_t>(alloc_type);
        current_[at] -= nbytes;
        --num_current_allocs_[at];
    }

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
    constexpr ScopedMemoryRecord& add_subscope(ScopedMemoryRecord const& subscope) {
        highest_peak_ = std::max(highest_peak_, current() + subscope.highest_peak_);
        for (AllocType type : {AllocType::PRIMARY, AllocType::FALLBACK}) {
            auto i = static_cast<std::size_t>(type);
            peak_[i] = std::max(peak_[i], current_[i] + subscope.peak_[i]);
            num_total_allocs_[i] += subscope.num_total_allocs_[i];
            num_current_allocs_[i] += subscope.num_current_allocs_[i];
            current_[i] += subscope.current_[i];
            total_[i] += subscope.total_[i];
        }
        return *this;
    }

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
    constexpr ScopedMemoryRecord& add_scope(ScopedMemoryRecord const& scope) {
        highest_peak_ = std::max(highest_peak_, scope.highest_peak_);
        for (AllocType type : {AllocType::PRIMARY, AllocType::FALLBACK}) {
            auto i = static_cast<std::size_t>(type);
            peak_[i] = std::max(peak_[i], scope.peak_[i]);
            current_[i] += scope.current_[i];
            total_[i] += scope.total_[i];
            num_total_allocs_[i] += scope.num_total_allocs_[i];
            num_current_allocs_[i] += scope.num_current_allocs_[i];
        }
        return *this;
    }

  private:
    AllocTypeArray num_current_allocs_{{0, 0}};
    AllocTypeArray num_total_allocs_{{0, 0}};
    AllocTypeArray current_{{0, 0}};
    AllocTypeArray total_{{0, 0}};
    AllocTypeArray peak_{{0, 0}};
    std::int64_t highest_peak_{0};

    /**
     * @brief Retrieves a value from a statistics array or accumulates the total.
     *
     * @param arr        The array containing statistics for each allocator type.
     * @param alloc_type The type of allocator to retrieve data for. If `AllocType::ALL`,
     *                   the function returns the sum across all entries in the array.
     * @return The requested statistic value or the accumulated total.
     */
    static constexpr std::int64_t get_or_accumulate(
        AllocTypeArray const& arr, AllocType alloc_type
    ) noexcept {
        if (alloc_type == AllocType::ALL) {
            return std::accumulate(arr.begin(), arr.end(), std::int64_t{0});
        }
        return arr[static_cast<std::size_t>(alloc_type)];
    }
};

static_assert(
    std::is_trivially_copyable_v<ScopedMemoryRecord>,
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
     * @brief Returns a copy of the main memory record.
     *
     * The main record tracks memory statistics for the lifetime of the resource.
     *
     * @return A copy of the current main memory record.
     */
    [[nodiscard]] ScopedMemoryRecord get_main_record() const;

    /**
     * @brief Get the total current allocated memory from both primary and fallback.
     *
     * @return Total number of currently allocated bytes.
     */
    [[nodiscard]] std::int64_t current_allocated() const noexcept;


    /**
     * @brief Begin recording a new scoped memory usage record for the current thread.
     *
     * This method pushes a new empty `ScopedMemoryRecord` onto the thread-local
     * record stack, allowing for nested memory tracking scopes.
     *
     * Must be paired with a matching call to `end_scoped_memory_record()`.
     *
     * @see end_scoped_memory_record()
     */
    void begin_scoped_memory_record();

    /**
     * @brief End the current scoped memory record and return it.
     *
     * Pops the top `ScopedMemoryRecord` from the thread-local stack and returns it.
     * If this scope was nested within another (i.e. if `begin_scoped_memory_record()` was
     * called multiple times in a row), the returned scope is automatically added as a
     * subscope to the next scope remaining on the stack.
     *
     * This allows nesting of scoped memory tracking, where each scope can contain one or
     * more subscopes. When analyzing or reporting memory statistics, the memory usage
     * of each scope can be calculated **inclusive of its subscopes**. This behavior
     * mimics standard hierarchical memory profilers, where the total memory attributed to
     * a scope includes all allocations made within it, plus those made in its nested
     * regions.
     *
     * @return The scope that was just ended.
     *
     * @throws std::out_of_range if called without a matching
     * `begin_scoped_memory_record()`.
     *
     * @see begin_scoped_memory_record()
     */
    ScopedMemoryRecord end_scoped_memory_record();

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

    /// Tracks memory statistics for the lifetime of the resource.
    ScopedMemoryRecord main_record_;
    /// Per-thread stack of scoped records, used with begin/end scoped memory tracking.
    std::unordered_map<std::thread::id, std::stack<ScopedMemoryRecord>> record_stacks_;
    /// Maps allocated memory pointers to the thread IDs that allocated them.
    std::unordered_map<void*, std::thread::id> allocating_threads_;
};


}  // namespace rapidsmpf
