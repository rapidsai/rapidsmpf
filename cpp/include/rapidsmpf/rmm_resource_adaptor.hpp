/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

#include <cuda/memory_resource>

#include <rmm/resource_ref.hpp>

#include <rapidsmpf/detail/rmm_resource_adaptor_impl.hpp>
#include <rapidsmpf/memory/scoped_memory_record.hpp>

namespace rapidsmpf {

/**
 * @brief A RMM memory resource adaptor tailored to RapidsMPF.
 *
 * This adaptor implements:
 * - Memory usage tracking.
 * - Fallback memory resource support upon out-of-memory in the primary resource.
 *
 * This class is copyable and shares ownership of its internal state via
 * `cuda::mr::shared_resource`.
 */
class RmmResourceAdaptor
    : public cuda::mr::shared_resource<detail::RmmResourceAdaptorImpl<
          cuda::mr::any_resource<cuda::mr::device_accessible>>> {
    using any_device_resource = cuda::mr::any_resource<cuda::mr::device_accessible>;
    using shared_base =
        cuda::mr::shared_resource<detail::RmmResourceAdaptorImpl<any_device_resource>>;

  public:
    /// @brief Tag this resource as device-accessible for the CCCL concept.
    friend void get_property(
        RmmResourceAdaptor const&, cuda::mr::device_accessible
    ) noexcept {}

    /**
     * @brief Construct with specified primary and optional fallback memory resource.
     *
     * @param primary_mr The primary memory resource.
     * @param fallback_mr Optional fallback memory resource.
     */
    RmmResourceAdaptor(
        cuda::mr::any_resource<cuda::mr::device_accessible> primary_mr,
        std::optional<cuda::mr::any_resource<cuda::mr::device_accessible>> fallback_mr =
            std::nullopt
    );

    ~RmmResourceAdaptor() = default;

    /**
     * @brief Equality comparison.
     *
     * Two adaptors are equal if and only if they share the same underlying shared state.
     *
     * @param other The other adaptor to compare.
     * @return True if both adaptors refer to the same shared resource instance.
     */
    [[nodiscard]] bool operator==(RmmResourceAdaptor const& other) const noexcept {
        return get() == other.get();
    }

    /**
     * @brief Get a reference to the primary upstream resource.
     *
     * @return Reference to the RMM memory resource.
     */
    [[nodiscard]] rmm::device_async_resource_ref get_upstream_resource() const noexcept;

    /**
     * @brief Get a reference to the fallback upstream resource.
     *
     * This resource is used if the primary resource throws `rmm::out_of_memory`.
     *
     * @return Optional reference to the fallback RMM memory resource.
     */
    [[nodiscard]] std::optional<rmm::device_async_resource_ref>
    get_fallback_resource() const noexcept;

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
};

static_assert(cuda::mr::resource_with<RmmResourceAdaptor, cuda::mr::device_accessible>);

}  // namespace rapidsmpf
