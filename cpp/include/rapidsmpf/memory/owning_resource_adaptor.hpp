/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <utility>

#include <cuda/memory_resource>
#include <cuda/stream_ref>

#include <rmm/aligned.hpp>

namespace rapidsmpf {

/**
 * @brief CCCL-compatible memory resource adaptor that wraps another
 * resource and carries additional owned state.
 *
 * Allocation, deallocation, and property queries are forwarded to the
 * wrapped resource. The adaptor additionally stores an arbitrary
 * value-semantic object whose lifetime is tied to the adaptor itself.
 *
 * This is useful when a resource must keep related state alive for as long
 * as the resource object exists, for example shared ownership of runtime
 * state, pools, contexts, or other resources required by the wrapped
 * allocator implementation.
 *
 * Properties exposed by the wrapped resource (via `get_property` ADL) are
 * forwarded transparently through `cuda::forward_property`.
 *
 * @tparam Resource The wrapped CCCL-compatible resource type. Must satisfy
 * the `cuda::mr::resource` concept and be copyable.
 * @tparam Owner The owned state type carried alongside the resource.
 */
template <typename Resource, typename BackRef>
class OwningResourceAdaptor
    : public cuda::forward_property<OwningResourceAdaptor<Resource, BackRef>, Resource> {
  public:
    /**
     * @brief Construct an owning wrapper.
     *
     * @param resource The wrapped resource (typically a refcount-shared handle, e.g.
     * `RmmResourceAdaptor`, copied cheaply).
     * @param back_ref Owning back-reference kept alive for the lifetime of the wrapper.
     */
    OwningResourceAdaptor(Resource resource, BackRef back_ref) noexcept
        : resource_{std::move(resource)}, back_ref_{std::move(back_ref)} {}

    /**
     * @brief Accessor used by `cuda::forward_property` to query properties on the wrapped
     * resource.
     *
     * @return Reference to the wrapped resource.
     */
    [[nodiscard]] Resource& get() noexcept {
        return resource_;
    }

    /** @copydoc get() */
    [[nodiscard]] Resource const& get() const noexcept {
        return resource_;
    }

    /**
     * @brief Allocate memory on a CUDA stream.
     *
     * Forwards the allocation request to the wrapped resource.
     *
     * @param stream CUDA stream associated with the allocation.
     * @param bytes Number of bytes to allocate.
     * @param alignment Required allocation alignment.
     * @return Pointer to the allocated memory.
     */
    void* allocate(
        cuda::stream_ref stream,
        std::size_t bytes,
        std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT
    ) {
        return resource_.allocate(stream, bytes, alignment);
    }

    /**
     * @brief Deallocate memory on a CUDA stream.
     *
     * Forwards the deallocation request to the wrapped resource.
     *
     * @param stream CUDA stream associated with the deallocation.
     * @param ptr Pointer to the memory to deallocate.
     * @param bytes Number of bytes originally allocated.
     * @param alignment Alignment of the original allocation.
     */
    void deallocate(
        cuda::stream_ref stream,
        void* ptr,
        std::size_t bytes,
        std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT
    ) noexcept {
        resource_.deallocate(stream, ptr, bytes, alignment);
    }

    /**
     * @brief Allocate memory synchronously.
     *
     * Forwards the allocation request to the wrapped resource.
     *
     * @param bytes Number of bytes to allocate.
     * @param alignment Required allocation alignment.
     * @return Pointer to the allocated memory.
     */
    void* allocate_sync(
        std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT
    ) {
        return resource_.allocate_sync(bytes, alignment);
    }

    /**
     * @brief Deallocate memory synchronously.
     *
     * Forwards the deallocation request to the wrapped resource.
     *
     * @param ptr Pointer to the memory to deallocate.
     * @param bytes Number of bytes originally allocated.
     * @param alignment Alignment of the original allocation.
     */
    void deallocate_sync(
        void* ptr,
        std::size_t bytes,
        std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT
    ) noexcept {
        resource_.deallocate_sync(ptr, bytes, alignment);
    }

    /**
     * @brief Compare wrapped resource identity.
     *
     * Two adaptors compare equal if their wrapped resources compare equal.
     *
     * @param other Adaptor to compare against.
     * @return `true` if both adaptors refer to the same underlying resource.
     */
    [[nodiscard]] bool operator==(OwningResourceAdaptor const& other) const noexcept {
        return resource_ == other.resource_;
    }

  private:
    Resource resource_;
    BackRef back_ref_;
};

}  // namespace rapidsmpf
