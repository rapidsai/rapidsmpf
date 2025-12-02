/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <rapidsmpf/error.hpp>

namespace rapidsmpf {

/**
 * @brief Memory resource that allocate host memory through malloc/free.
 */
class HostMemoryResource {
  public:
    HostMemoryResource() = default;
    virtual ~HostMemoryResource() = default;
    HostMemoryResource(HostMemoryResource const&) = default;  ///< Copyable.
    HostMemoryResource(HostMemoryResource&&) = default;  ///< Movable.
    /// @brief Move assignment @returns Moved this.
    HostMemoryResource& operator=(HostMemoryResource const&) = default;
    /// @brief Copy assignment @returns Copied this.
    HostMemoryResource& operator=(HostMemoryResource&&) = default;

  /**
   * @brief Allocates memory of size at least \p bytes on the specified stream.
   *
   * The returned pointer will have 256 byte alignment regardless of the value
   * of alignment. Higher alignments must use the aligned_resource_adaptor.
   *
   * @throws rmm::bad_alloc When the requested `bytes` cannot be allocated.
   *
   * @param stream The stream on which to perform the allocation
   * @param bytes The size of the allocation
   * @param alignment The alignment of the allocation (see notes above)
   * @return void* Pointer to the newly allocated memory
   */
  void* allocate(rmm::cuda_stream_view stream,
                 std::size_t bytes,
                 std::size_t alignment = 1)
  {
    RAPIDSMPF_EXPECTS(alignment == 1, "cannot guarantee alignment", std::invalid_argument);
    return do_allocate(bytes, stream);
  }

  /**
   * @brief Deallocate memory pointed to by \p ptr on the specified stream.
   *
   * @param stream The stream on which to perform the deallocation
   * @param ptr Pointer to be deallocated
   * @param bytes The size in bytes of the allocation. This must be equal to the
   * value of `bytes` that was passed to the `allocate` call that returned `p`.
   * @param alignment The alignment that was passed to the `allocate` call that returned `p`
   */
  void deallocate(rmm::cuda_stream_view stream,
                  void* ptr,
                  std::size_t bytes,
                  [[maybe_unused]] std::size_t alignment = 1) noexcept
  {
    do_deallocate(ptr, bytes, stream);
  }

  /**
   * @brief Compare this resource to another.
   *
   * Two device_memory_resources compare equal if and only if memory allocated
   * from one device_memory_resource can be deallocated from the other and vice
   * versa.
   *
   * By default, simply checks if \p *this and \p other refer to the same
   * object, i.e., does not check if they are two objects of the same class.
   *
   * @param other The other resource to compare to
   * @returns If the two resources are equivalent
   */
  [[nodiscard]] bool is_equal(HostMemoryResource const& other) const noexcept
  {
    return do_is_equal(other);
  }

  /**
   * @brief Comparison operator with another device_memory_resource
   *
   * @param other The other resource to compare to
   * @return true If the two resources are equivalent
   * @return false If the two resources are not equivalent
   */
  [[nodiscard]] bool operator==(HostMemoryResource const& other) const noexcept
  {
    return do_is_equal(other);
  }

  /**
   * @brief Comparison operator with another HostMemoryResource
   *
   * @param other The other resource to compare to
   * @return false If the two resources are equivalent
   * @return true If the two resources are not equivalent
   */
  [[nodiscard]] bool operator!=(HostMemoryResource const& other) const noexcept
  {
    return !do_is_equal(other);
  }

  private:
    /**
     * @brief Allocates memory of size at least \p size.
     *
     * The stream argument is ignored.
     *
     * @param size The size of the allocation
     * @param stream This argument is ignored
     * @return void* Pointer to the newly allocated memory
     */
    virtual void* do_allocate(
        std::size_t size, [[maybe_unused]] rmm::cuda_stream_view stream
    ) {
        return ::operator new(size);
    }

    /**
     * @brief Deallocate memory pointed to by \p ptr.
     *
     * This function synchronizes the stream before deallocating the memory.
     *
     * @param ptr Pointer to be deallocated
     * @param size The size in bytes of the allocation. This must be equal to the value
     * of `bytes` that was passed to the `allocate` call that returned `ptr`.
     * @param stream The stream in which to order this deallocation
     */
    virtual void do_deallocate(
        void* ptr, [[maybe_unused]] std::size_t size, rmm::cuda_stream_view stream
    ) noexcept {
        // Since `free` is immediate (not stream ordered), we need to wait for
        // in-flight CUDA operations to finish before freeing the memory, to
        // avoid potential use-after-free errors or race conditions.
        stream.synchronize();
        ::operator delete(ptr);
    }


    /**
     * @brief Compare this resource to another.
     *
     * Two system_memory_resources always compare equal, because they can each deallocate
     * memory allocated by the other.
     *
     * @param other The other resource to compare to
     * @return true If the two resources are equivalent
     * @return false If the two resources are not equal
     */
    [[nodiscard]] virtual bool do_is_equal(
        HostMemoryResource const& other
    ) const noexcept {
        return this == &other;
    }

    /**
     * @brief Enables the `cuda::mr::host_accessible` property
     *
     * This property declares that a `HostMemoryResource` provides host-accessible memory
     */
    friend void get_property(
        HostMemoryResource const&, cuda::mr::host_accessible
    ) noexcept {}
};

static_assert(cuda::mr::resource<HostMemoryResource>);
static_assert(cuda::mr::resource_with<HostMemoryResource, cuda::mr::host_accessible>);
static_assert(!cuda::mr::resource_with<HostMemoryResource, cuda::mr::device_accessible>);

}  // namespace rapidsmpf
