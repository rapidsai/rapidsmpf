/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>

#include <rmm/aligned.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <rapidsmpf/error.hpp>

namespace rapidsmpf {

/**
 * @brief Host memory resource using standard CPU allocation.
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
class HostMemoryResource {
  public:
    /// @brief Default constructor.
    HostMemoryResource() = default;

    /// @brief Virtual destructor to allow polymorphic use.
    virtual ~HostMemoryResource() = default;

    HostMemoryResource(HostMemoryResource const&) = default;  ///< Copyable.
    HostMemoryResource(HostMemoryResource&&) = default;  ///< Movable.

    /// @brief Copy assignment.
    /// @return Reference to this object after assignment.
    HostMemoryResource& operator=(HostMemoryResource const&) = default;

    /// @brief Move assignment.
    /// @return Reference to this object after assignment.
    HostMemoryResource& operator=(HostMemoryResource&&) = default;

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
     * Derived classes may override this to provide custom host allocation strategies.
     *
     * @param stream CUDA stream associated with the allocation.
     * @param size Number of bytes to at least allocate.
     * @param alignment Required alignment.
     * @return Pointer to the allocated memory.
     *
     * @throw std::bad_alloc If the allocation fails.
     * @throw std::invalid_argument If @p alignment is not a valid alignment.
     */
    virtual void* allocate(
        rmm::cuda_stream_view stream,
        std::size_t size,
        std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT
    );

    /**
     * @brief Deallocates host memory associated with a CUDA stream.
     *
     * The default implementation synchronizes @p stream before deallocating the
     * memory with the ``delete`` operator. This ensures that any in-flight CUDA
     * operations using the memory complete before it is freed.
     *
     * Derived classes may override this to provide custom host deallocation strategies.
     *
     * @param stream CUDA stream associated with operations that used @p ptr.
     * @param ptr Pointer to the memory to deallocate. May be nullptr.
     * @param size Number of bytes previously allocated at @p ptr.
     * @param alignment Alignment originally used for the allocation.
     */
    virtual void deallocate(
        rmm::cuda_stream_view stream,
        void* ptr,
        std::size_t size,
        std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT
    ) noexcept;

    /**
     * @brief Compares this resource to another resource.
     *
     * Two resources are considered equal if memory allocated by one may be
     * deallocated by the other. The default implementation compares object identity.
     *
     * The base class is stateless, and all instances behave identically. Any
     * instance can deallocate memory allocated by any other instance of this
     * base class, so the comparison always returns true.
     *
     * Derived classes that use different allocation or deallocation strategies
     * must override this function. Such classes should return true only when
     * the other resource is compatible with their allocation and free methods.
     *
     * @param other The resource to compare with.
     * @return true because all instances of this base class are considered equal.
     */
    [[nodiscard]] virtual bool is_equal(
        [[maybe_unused]] HostMemoryResource const& other
    ) const noexcept {
        return true;
    }

    /// @copydoc is_equal()
    [[nodiscard]] bool operator==(HostMemoryResource const& other) const noexcept {
        return is_equal(other);
    }

    /// @copydoc is_equal()
    [[nodiscard]] bool operator!=(HostMemoryResource const& other) const noexcept {
        return !is_equal(other);
    }

    /**
     * @brief Enables the `cuda::mr::host_accessible` property
     *
     * This property declares that a `HostMemoryResource` provides host accessible memory
     */
    friend void get_property(
        HostMemoryResource const&, cuda::mr::host_accessible
    ) noexcept {}

    /**
     * @brief Enables the `cuda::mr::async` property
     *
     * This property declares that a `HostMemoryResource` supports async operations
     */
    friend void get_property(HostMemoryResource const&, cuda::mr::async) noexcept {}
};

static_assert(cuda::mr::resource<HostMemoryResource>);
static_assert(cuda::mr::resource_with<HostMemoryResource, cuda::mr::host_accessible>);
static_assert(!cuda::mr::resource_with<HostMemoryResource, cuda::mr::device_accessible>);

}  // namespace rapidsmpf
