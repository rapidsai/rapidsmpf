/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
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
 * This resource allocates host memory using the ``new`` and ``delete`` operator. It is
 * intended for use with `cuda::mr::resource` and related facilities, and advertises the
 * `cuda::mr::host_accessible` property.
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
     * @brief Synchronously allocates host memory.
     *
     * @param size Number of bytes to allocate.
     * @param alignment Required alignment, must be a power of two.
     * @return Pointer to the allocated memory.
     *
     * @throw std::bad_alloc If the allocation fails.
     * @throw std::invalid_argument If @p alignment is not a valid alignment.
     */
    void* allocate_sync(std::size_t size, std::size_t alignment) {
        return do_allocate(size, rmm::cuda_stream_default, alignment);
    }

    /**
     * @brief Synchronously deallocates host memory.
     *
     * The CUDA default stream is synchronized before deallocation to ensure
     * that any in-flight operations that might touch @p ptr have completed.
     *
     * @param ptr Pointer to the memory to deallocate. May be nullptr.
     * @param size Number of bytes previously allocated at @p ptr.
     * @param alignment Alignment originally used for the allocation.
     */
    void deallocate_sync(
        void* ptr, std::size_t size, [[maybe_unused]] std::size_t alignment
    ) noexcept {
        do_deallocate(ptr, size, rmm::cuda_stream_default, alignment);
    }

    /**
     * @brief Allocates host memory associated with a CUDA stream.
     *
     * @param stream CUDA stream associated with the allocation.
     * @param bytes Number of bytes to allocate.
     * @param alignment Required alignment.
     * @return Pointer to the allocated memory.
     *
     * @throw std::bad_alloc If the allocation fails.
     * @throw std::invalid_argument If @p alignment is not a valid alignment.
     */
    void* allocate(
        rmm::cuda_stream_view stream,
        std::size_t bytes,
        std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT
    ) {
        return do_allocate(bytes, stream, alignment);
    }

    /**
     * @brief Deallocates host memory associated with a CUDA stream.
     *
     * @param stream CUDA stream associated with operations that used @p ptr.
     * @param ptr Pointer to the memory to deallocate. May be nullptr.
     * @param size Number of bytes previously allocated at @p ptr.
     * @param alignment Alignment originally used for the allocation.
     */
    void deallocate(
        rmm::cuda_stream_view stream,
        void* ptr,
        std::size_t size,
        [[maybe_unused]] std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT
    ) noexcept {
        do_deallocate(ptr, size, stream, alignment);
    }

    /**
     * @brief Compare this resource to another.
     *
     * Two resources are considered equal if memory allocated by one may be
     * deallocated by the other. The default implementation compares object identity.
     *
     * @param other Resource to compare with.
     * @return true if the two resources are equal, false otherwise.
     */
    [[nodiscard]] bool is_equal(HostMemoryResource const& other) const noexcept {
        return do_is_equal(other);
    }

    /// @copydoc is_equal()
    [[nodiscard]] bool operator==(HostMemoryResource const& other) const noexcept {
        return do_is_equal(other);
    }

    /// @copydoc is_equal()
    [[nodiscard]] bool operator!=(HostMemoryResource const& other) const noexcept {
        return !do_is_equal(other);
    }

  private:
    /**
     * @brief Allocates memory of size at least @p size with the given alignment.
     *
     * Derived classes may override this to provide custom host allocation strategies.
     *
     * @param size Number of bytes to allocate.
     * @param stream CUDA stream associated with the allocation.
     * @param alignment Required alignment.
     *
     * @return Pointer to the allocated memory.
     *
     * @throw std::bad_alloc If the allocation fails.
     */
    virtual void* do_allocate(
        std::size_t size,
        [[maybe_unused]] rmm::cuda_stream_view stream,
        std::size_t alignment
    ) {
        return ::operator new(size, std::align_val_t{alignment});
    }

    /**
     * @brief Deallocates memory using the given alignment.
     *
     * The default implementation synchronizes @p stream before deallocating the
     * memory with the ``delete`` operator. This ensures that any in-flight CUDA
     * operations using the memory complete before it is freed.
     *
     * @param ptr Pointer to the memory to deallocate.
     * @param size Number of bytes previously allocated.
     * @param stream CUDA stream associated with operations using @p ptr.
     * @param alignment Alignment used when allocating @p ptr.
     */
    virtual void do_deallocate(
        void* ptr,
        [[maybe_unused]] std::size_t size,
        rmm::cuda_stream_view stream,
        std::size_t alignment
    ) noexcept {
        stream.synchronize();
        ::operator delete(ptr, std::align_val_t{alignment});
    }

    /**
     * @brief Compares this resource to another.
     *
     * The default implementation considers resources equal only if they are
     * the same object. Derived implementations may override this to permit
     * equivalence between different instances.
     *
     * @param other Resource to compare with.
     * @return true if the resources are considered equal, false otherwise.
     */
    [[nodiscard]] virtual bool do_is_equal(
        HostMemoryResource const& other
    ) const noexcept {
        return this == &other;
    }

    /**
     * @brief Enables the `cuda::mr::host_accessible` property
     *
     * This property declares that a `HostMemoryResource` provides host accessible memory
     */
    friend void get_property(
        HostMemoryResource const&, cuda::mr::host_accessible
    ) noexcept {}
};

static_assert(cuda::mr::resource<HostMemoryResource>);
static_assert(cuda::mr::resource_with<HostMemoryResource, cuda::mr::host_accessible>);
static_assert(!cuda::mr::resource_with<HostMemoryResource, cuda::mr::device_accessible>);

}  // namespace rapidsmpf
