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
 * @brief A RMM memory resource adaptor tailored to RapidsMPF.
 *
 * This resource implements a broad range of features used throughout RapidsMPF.
 *
 * Memory Statistics
 * -----------------
 *
 *
 *
 * Alternate on OOM
 * ----------------
 * If set, uses an fallback upstream resource when the primary upstream resource throws
 * `rmm::out_of_memory`.
 *
 *
 */
class RmmResourceAdaptor final : public rmm::mr::device_memory_resource {
  public:
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
     * @brief Get a reference to the alternative upstream resource.
     *
     * This resource is used when primary upstream resource throws `rmm::out_of_memory`.
     *
     * @return Reference to the RMM memory resource.
     */
    [[nodiscard]] std::optional<rmm::device_async_resource_ref> get_fallback_resource(
    ) const noexcept {
        return fallback_mr_;
    }

  private:
    /**
     * @brief Allocates memory of size at least `bytes` using the upstream
     * resource.
     *
     * @throws any exceptions thrown from the upstream resources, only
     * `rmm::out_of_memory` thrown by the primary upstream is caught.
     *
     * @param bytes The size, in bytes, of the allocation
     * @param stream Stream on which to perform the allocation
     * @return void* Pointer to the newly allocated memory
     */
    void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) override;

    /**
     * @brief Free allocation of size `bytes` pointed to by `ptr`
     *
     * @param ptr Pointer to be deallocated
     * @param bytes Size of the allocation
     * @param stream Stream on which to perform the deallocation
     */
    void do_deallocate(void* ptr, std::size_t bytes, rmm::cuda_stream_view stream)
        override;

    /**
     * @brief Compare the resource to another.
     *
     * @param other The other resource to compare to
     * @return true If the two resources are equivalent
     * @return false If the two resources are not equal
     */
    [[nodiscard]] bool do_is_equal(rmm::mr::device_memory_resource const& other
    ) const noexcept override;

    std::mutex mutex_;
    rmm::device_async_resource_ref primary_mr_;
    std::optional<rmm::device_async_resource_ref> fallback_mr_;
    std::unordered_set<void*> fallback_allocations_;
};


}  // namespace rapidsmpf
