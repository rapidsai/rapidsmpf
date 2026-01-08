/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <mutex>
#include <optional>
#include <stack>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <rmm/error.hpp>
#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <rapidsmpf/memory/scoped_memory_record.hpp>

namespace rapidsmpf {

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
        : primary_mr_{std::move(primary_mr)}, fallback_mr_{std::move(fallback_mr)} {}

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
    [[nodiscard]] std::optional<rmm::device_async_resource_ref>
    get_fallback_resource() const noexcept {
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
    void do_deallocate(
        void* ptr, std::size_t bytes, rmm::cuda_stream_view stream
    ) noexcept override;

    /**
     * @brief Check if this memory resource is equal to another.
     *
     * Equality is defined by comparing the primary and fallback resources.
     *
     * @param other Another memory resource to compare against.
     * @return true if both resources are equivalent, false otherwise.
     */
    [[nodiscard]] bool do_is_equal(
        rmm::mr::device_memory_resource const& other
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
