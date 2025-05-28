/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

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
    std::uint64_t num_calls{0};  ///< Number of times the scope was executed.
    std::uint64_t num_allocs{0};  ///< Total number of allocations in the scope.
    std::uint64_t current{0};  ///< Current memory allocated in the scope (bytes).
    std::uint64_t total{0};  ///< Total memory allocated in the scope (bytes).
    std::uint64_t peak{0};  ///< Peak memory usage in the scope (bytes).

    /**
     * @brief Record a memory allocation.
     *
     * Updates allocation counters and peak memory usage.
     *
     * @param nbytes Number of bytes allocated.
     */
    void record_allocation(std::uint64_t nbytes) {
        ++num_allocs;
        current += nbytes;
        total += nbytes;
        peak = std::max(nbytes, peak);
    }

    /**
     * @brief Record a memory deallocation.
     *
     * Updates the current memory usage.
     *
     * @param nbytes Number of bytes deallocated.
     */
    void record_deallocation(std::uint64_t nbytes) {
        current -= nbytes;
    }
};

/**
 * @brief A RMM memory resource adaptor tailored to RapidsMPF.
 *
 * This adaptor implements:
 * - Memory usage tracking (current, total, peak).
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
    ScopedMemoryRecord primary_record_;
    ScopedMemoryRecord fallback_record_;
};


}  // namespace rapidsmpf
