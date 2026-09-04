/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>

#include <cuda/stream>

#include <rmm/aligned.hpp>
#include <rmm/resource_ref.hpp>

#include <rapidsmpf/detail/rmm_resource_adaptor_impl.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/back_ref_mixin.hpp>
#include <rapidsmpf/memory/scoped_memory_record.hpp>

namespace rapidsmpf {

class BufferResource;

namespace detail {

/**
 * @brief Untracked host memory resource using standard CPU allocation.
 *
 * This resource allocates pageable host memory using the ``new`` and ``delete``
 * operators. It is intended for use with `cuda::mr::resource` and related
 * facilities, and advertises the `cuda::mr::host_accessible` property.
 *
 * For sufficiently large allocations (>4 MiB), this resource also issues a
 * best-effort request to enable Transparent Huge Pages (THP) on the allocated
 * region. THP can improve device-host memory transfer performance for large
 * buffers. The hint is applied via `madvise(MADV_HUGEPAGE)` and may be ignored
 * by the kernel depending on system configuration or resource availability.
 */
class HostMemoryResourceImpl {
  public:
    HostMemoryResourceImpl() = default;
    ~HostMemoryResourceImpl() = default;

    /// @brief Copy-construct a host memory resource.
    /// @param other Resource to copy.
    HostMemoryResourceImpl(HostMemoryResourceImpl const& other) = default;

    /// @brief Move-construct a host memory resource.
    /// @param other Resource to move.
    HostMemoryResourceImpl(HostMemoryResourceImpl&& other) = default;

    /// @brief Copy assignment.
    /// @return Reference to this object after assignment.
    HostMemoryResourceImpl& operator=(HostMemoryResourceImpl const&) = default;

    /// @brief Move assignment.
    /// @return Reference to this object after assignment.
    HostMemoryResourceImpl& operator=(HostMemoryResourceImpl&&) = default;

    /**
     * @brief Synchronously allocates host memory is disabled.
     *
     * Always use stream-ordered allocators in RapidsMPF.
     *
     * @return N/A.
     *
     * @throw std::invalid_argument Always.
     */
    void* allocate_sync(std::size_t, std::size_t) {
        RAPIDSMPF_FAIL(
            "only async stream-ordered allocation must be used in RapidsMPF",
            std::invalid_argument
        );
    }

    /**
     * @brief Synchronously deallocates host memory is disabled.
     *
     * @throw std::invalid_argument Always.
     */
    void deallocate_sync(void*, std::size_t, std::size_t) {
        RAPIDSMPF_FAIL(
            "only async stream-ordered allocation must be used in RapidsMPF",
            std::invalid_argument
        );
    }

    /**
     * @brief Allocates host memory associated with a CUDA stream.
     *
     * @param stream CUDA stream associated with the allocation.
     * @param size Number of bytes to at least allocate.
     * @param alignment Required alignment.
     * @return Pointer to the allocated memory.
     *
     * @throw std::bad_alloc If the allocation fails.
     * @throw std::invalid_argument If @p alignment is not a valid alignment.
     */
    void* allocate(
        cuda::stream_ref stream,
        std::size_t size,
        std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT
    );

    /**
     * @brief Deallocates host memory associated with a CUDA stream.
     *
     * Synchronizes @p stream before deallocating the memory with the ``delete``
     * operator.
     *
     * @param stream CUDA stream associated with operations that used @p ptr.
     * @param ptr Pointer to the memory to deallocate. May be nullptr.
     * @param size Number of bytes previously allocated at @p ptr.
     * @param alignment Alignment originally used for the allocation.
     */
    void deallocate(
        cuda::stream_ref stream,
        void* ptr,
        std::size_t size,
        std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT
    ) noexcept;

    /**
     * @brief Compares this resource to another resource.
     *
     * All instances are stateless and interchangeable, so this always returns
     * true.
     *
     * @param other The resource to compare with.
     * @return true
     */
    [[nodiscard]] bool operator==(
        [[maybe_unused]] HostMemoryResourceImpl const& other
    ) const noexcept {
        return true;
    }

    /**
     * @brief Enables the `cuda::mr::host_accessible` property.
     *
     * This property declares that a `HostMemoryResourceImpl` provides host-accessible
     * memory.
     */
    friend void get_property(
        HostMemoryResourceImpl const&, cuda::mr::host_accessible
    ) noexcept {}
};

static_assert(cuda::mr::resource<HostMemoryResourceImpl>);
static_assert(cuda::mr::resource_with<HostMemoryResourceImpl, cuda::mr::host_accessible>);
static_assert(
    !cuda::mr::resource_with<HostMemoryResourceImpl, cuda::mr::device_accessible>
);

}  // namespace detail

/**
 * @brief Shared, tracked pageable-host memory resource.
 *
 * Wraps the standard CPU allocator in `RmmResourceAdaptorImpl` so live
 * allocations and lifetime memory records are available to `BufferResource`.
 * Copies share allocation statistics and the upstream resource.
 */
class HostMemoryResource final
    : public cuda::mr::shared_resource<
          detail::RmmResourceAdaptorImpl<detail::HostMemoryResourceImpl>>,
      public BackRefMixin<BufferResource> {
    using shared_base = cuda::mr::shared_resource<
        detail::RmmResourceAdaptorImpl<detail::HostMemoryResourceImpl>>;

  public:
    ~HostMemoryResource() = default;

    /// @brief Copy-construct a shared host memory resource.
    /// @param other Resource to copy.
    HostMemoryResource(HostMemoryResource const& other) = default;

    /// @brief Move-construct a shared host memory resource.
    /// @param other Resource to move.
    HostMemoryResource(HostMemoryResource&& other) = default;

    /// @brief Copy-assign a shared host memory resource.
    /// @param other Resource to copy.
    /// @return Reference to this object after assignment.
    HostMemoryResource& operator=(HostMemoryResource const& other) = default;

    /// @brief Move-assign a shared host memory resource.
    /// @param other Resource to move.
    /// @return Reference to this object after assignment.
    HostMemoryResource& operator=(HostMemoryResource&& other) = default;

    /// @copydoc detail::HostMemoryResourceImpl::allocate_sync
    void* allocate_sync(std::size_t, std::size_t) {
        RAPIDSMPF_FAIL(
            "only async stream-ordered allocation must be used in RapidsMPF",
            std::invalid_argument
        );
    }

    /// @copydoc detail::HostMemoryResourceImpl::deallocate_sync
    void deallocate_sync(void*, std::size_t, std::size_t) {
        RAPIDSMPF_FAIL(
            "only async stream-ordered allocation must be used in RapidsMPF",
            std::invalid_argument
        );
    }

    /// @copydoc detail::HostMemoryResourceImpl::allocate
    void* allocate(
        cuda::stream_ref stream,
        std::size_t size,
        std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT
    ) {
        return get().allocate(stream, size, alignment);
    }

    /// @copydoc detail::HostMemoryResourceImpl::deallocate
    void deallocate(
        cuda::stream_ref stream,
        void* ptr,
        std::size_t size,
        std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT
    ) noexcept {
        get().deallocate(stream, ptr, size, alignment);
    }

    /// @copydoc detail::HostMemoryResourceImpl::operator==
    [[nodiscard]] bool operator==(HostMemoryResource const& other) const noexcept {
        return get() == other.get();
    }

    /// @copydoc RmmResourceAdaptor::current_allocated
    [[nodiscard]] std::int64_t current_allocated() const noexcept {
        return get().current_allocated();
    }

    /// @copydoc RmmResourceAdaptor::get_main_record
    [[nodiscard]] ScopedMemoryRecord get_main_memory_record() const {
        return get().get_main_record();
    }

  private:
    HostMemoryResource()
        : shared_base(
              cuda::mr::make_shared_resource<
                  detail::RmmResourceAdaptorImpl<detail::HostMemoryResourceImpl>>(
                  std::in_place
              )
          ) {}

    friend class BufferResource;
};

static_assert(cuda::mr::resource<HostMemoryResource>);
static_assert(cuda::mr::resource_with<HostMemoryResource, cuda::mr::host_accessible>);
static_assert(!cuda::mr::resource_with<HostMemoryResource, cuda::mr::device_accessible>);

}  // namespace rapidsmpf
