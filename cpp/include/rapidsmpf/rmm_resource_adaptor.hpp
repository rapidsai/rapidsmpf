/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda/memory_resource>

#include <rmm/resource_ref.hpp>

#include <rapidsmpf/detail/rmm_resource_adaptor_impl.hpp>
#include <rapidsmpf/memory/back_ref_mixin.hpp>
#include <rapidsmpf/memory/scoped_memory_record.hpp>

namespace rapidsmpf {
class BufferResource;

/**
 * @brief A RMM memory resource adaptor tailored to RapidsMPF.
 *
 * This adaptor wraps a primary device memory resource and adds memory usage
 * tracking (lifetime stats plus per-thread scoped records).
 *
 * This class is copyable and shares ownership of its internal state via
 * `cuda::mr::shared_resource`.
 *
 * The primary-resource constructor is private: instances can only be created by
 * `BufferResource`, which installs the back-reference before the adaptor becomes
 * observable. Obtain one via `BufferResource::device_mr_adaptor()` (copies of
 * which are valid). This guarantees the `BackRefMixin<BufferResource>` lifetime
 * contract always holds — every adaptor a caller can reach already has a
 * back-reference installed, so copies keep their owning `BufferResource` alive
 * for as long as any copy lives.
 */
class RmmResourceAdaptor
    : public cuda::mr::shared_resource<detail::RmmResourceAdaptorImpl<
          cuda::mr::any_resource<cuda::mr::device_accessible>>>,
      public BackRefMixin<BufferResource> {
    using any_device_resource = cuda::mr::any_resource<cuda::mr::device_accessible>;
    using shared_base =
        cuda::mr::shared_resource<detail::RmmResourceAdaptorImpl<any_device_resource>>;

  public:
    /// @brief Tag this resource as device-accessible for the CCCL concept.
    friend void get_property(
        RmmResourceAdaptor const&, cuda::mr::device_accessible
    ) noexcept {}

    ~RmmResourceAdaptor() = default;

    /**
     * @brief Equality comparison.
     *
     * Two adaptors are equal iff they share the same underlying shared
     * state **and** reference the same owning `BufferResource` (or are
     * both standalone).
     *
     * @param other The other adaptor to compare.
     * @return True if both adaptors share the same shared state and the
     * same owning `BufferResource`.
     */
    [[nodiscard]] bool operator==(RmmResourceAdaptor const& other) const noexcept {
        return get() == other.get() && BackRefMixin<BufferResource>::operator==(other);
    }

    /**
     * @brief Get a reference to the primary upstream resource.
     *
     * @return Reference to the RMM memory resource.
     */
    [[nodiscard]] rmm::device_async_resource_ref get_upstream_resource() const noexcept;

    /**
     * @brief Returns a copy of the main memory record.
     *
     * The main record tracks memory statistics for the lifetime of the resource.
     *
     * @return A copy of the current main memory record.
     */
    [[nodiscard]] ScopedMemoryRecord get_main_record() const;

    /**
     * @brief Get the total current allocated memory through this resource.
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
    // Only `BufferResource` may create the primary adaptor; all other instances
    // are copies obtained via `BufferResource::device_mr_adaptor()`. This keeps
    // the back-reference lifetime contract enforceable (see class docs).
    friend class BufferResource;

    /**
     * @brief Construct with the specified primary memory resource.
     *
     * @param primary_mr The primary memory resource.
     */
    explicit RmmResourceAdaptor(
        cuda::mr::any_resource<cuda::mr::device_accessible> primary_mr
    );
};

static_assert(cuda::mr::resource_with<RmmResourceAdaptor, cuda::mr::device_accessible>);

}  // namespace rapidsmpf
