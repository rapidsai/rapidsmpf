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
 * @brief Memory statistics for a specific scope.
 */
struct ScopedMemoryRecord {
    /// Allocation source types.
    enum class AllocType : std::size_t {
        Primary = 0,
        Fallback = 1
    };

    /**
     * @brief Returns the number of allocations for the specified allocator type.
     * @param alloc_type The allocator type.
     * @return Number of allocations for the given allocator type.
     */
    [[nodiscard]] std::uint64_t num_allocs(AllocType alloc_type) const noexcept {
        return num_allocs_[static_cast<std::size_t>(alloc_type)];
    }

    /**
     * @brief Returns the current memory usage (bytes) for the specified allocator type.
     * @param alloc_type The allocator type.
     * @return Current memory usage in bytes for the given allocator type.
     */
    [[nodiscard]] std::uint64_t current(AllocType alloc_type) const noexcept {
        return current_[static_cast<std::size_t>(alloc_type)];
    }

    /**
     * @brief Returns the total memory allocated (bytes) for the specified allocator type.
     * @param alloc_type The allocator type.
     * @return Total memory allocated in bytes for the given allocator type.
     */
    [[nodiscard]] std::uint64_t total(AllocType alloc_type) const noexcept {
        return total_[static_cast<std::size_t>(alloc_type)];
    }

    /**
     * @brief Returns the peak memory usage (bytes) for the specified allocator type.
     * @param alloc_type The allocator type.
     * @return Peak memory usage in bytes for the given allocator type.
     */
    [[nodiscard]] std::uint64_t peak(AllocType alloc_type) const noexcept {
        return peak_[static_cast<std::size_t>(alloc_type)];
    }

    /**
     * @brief Returns the highest combined memory usage (bytes).
     * @return Highest combined memory usage in bytes.
     */
    [[nodiscard]] std::uint64_t highest_peak() const noexcept {
        return highest_peak_;
    }

    /**
     * @brief Returns the number of times the scope was executed.
     * @return Number of times the scope was executed.
     */
    [[nodiscard]] std::uint64_t num_calls() const noexcept {
        return num_calls_;
    }

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
        ++num_allocs_[at];
        current_[at] += nbytes;
        total_[at] += nbytes;
        peak_[at] = std::max(peak_[at], current_[at]);
        highest_peak_ = std::max(highest_peak_, current_[at]);
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
        current_[at] -= nbytes;
    }

  private:
    std::uint64_t num_calls_{0};
    std::array<std::uint64_t, 2> num_allocs_{{0, 0}};
    std::array<std::uint64_t, 2> current_{{0, 0}};
    std::array<std::uint64_t, 2> total_{{0, 0}};
    std::array<std::uint64_t, 2> peak_{{0, 0}};
    std::uint64_t highest_peak_{0};
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
