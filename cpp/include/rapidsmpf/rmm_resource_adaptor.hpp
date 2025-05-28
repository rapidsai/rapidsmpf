/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstddef>
#include <mutex>
#include <optional>
#include <unordered_set>

#include <rmm/error.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

namespace rapidsmpf {
/**
 * @brief Tracks memory statistics for a specific scope.
 */
struct ScopedMemoryRecord {
    /// Allocation source types.
    enum class AllocType : std::size_t {
        Primary = 0,
        Fallback = 1
    };

    /// Number of times the scope was executed.
    std::uint64_t num_calls{0};

    /// Number of allocations by allocator type.
    std::array<std::uint64_t, 2> num_allocs = {0};

    /// Current memory usage (bytes) by allocator type.
    std::array<std::uint64_t, 2> current = {0};

    /// Total memory allocated (bytes) by allocator type.
    std::array<std::uint64_t, 2> total = {0};

    /// Peak memory usage (bytes) by allocator type.
    std::array<std::uint64_t, 2> peak = {0};

    /// Highest combined memory usage (bytes).
    std::uint64_t highest_peak = {0};

    /**
     * @brief Records a memory allocation.
     *
     * Increments allocation counters, updates current and total allocations, and tracks
     * peak usage.
     *
     * @param alloc_type The allocator type.
     * @param nbytes     Number of bytes allocated.
     */
    void record_allocation(AllocType alloc_type, std::uint64_t nbytes) {
        auto at = static_cast<std::size_t>(alloc_type);
        ++num_allocs[at];
        current[at] += nbytes;
        total[at] += nbytes;
        peak[at] = std::max(peak[at], nbytes);
        highest_peak = std::max(highest_peak, nbytes);
    }

    /**
     * @brief Records a memory deallocation event.
     *
     * Decreases the current memory usage for the specified allocator type.
     *
     * @param alloc_type The allocator type.
     * @param nbytes     Number of bytes deallocated.
     */
    void record_deallocation(AllocType alloc_type, std::uint64_t nbytes) {
        auto at = static_cast<std::size_t>(alloc_type);
        current[at] -= nbytes;
    }
};

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
     * @brief Get the total current allocated memory from both primary and fallback.
     *
     * @return Total number of currently allocated bytes.
     */
    [[nodiscard]] ScopedMemoryRecord const& get_record() const noexcept {
        return record_;
    }

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
    ScopedMemoryRecord record_;
};


}  // namespace rapidsmpf
