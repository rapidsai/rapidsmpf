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
 * @brief `device_memory_resource` derived class that uses malloc/free.
 */
class HostMemoryResource final : public rmm::mr::device_memory_resource {
  public:
    HostMemoryResource() = default;
    ~HostMemoryResource() override = default;
    HostMemoryResource(HostMemoryResource const&) = default;  ///< Copyable.
    HostMemoryResource(HostMemoryResource&&) = default;  ///< Movable.
    /// @brief Move assignment @returns Moved this.
    HostMemoryResource& operator=(HostMemoryResource const&) = default;
    /// @brief Copy assignment @returns Copied this.
    HostMemoryResource& operator=(HostMemoryResource&&) = default;

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
    void* do_allocate(
        std::size_t size, [[maybe_unused]] rmm::cuda_stream_view stream
    ) override {
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
    void do_deallocate(
        void* ptr, [[maybe_unused]] std::size_t size, rmm::cuda_stream_view stream
    ) noexcept override {
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
    [[nodiscard]] bool do_is_equal(
        device_memory_resource const& other
    ) const noexcept override {
        return dynamic_cast<HostMemoryResource const*>(&other) != nullptr;
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

static_assert(cuda::mr::resource_with<HostMemoryResource, cuda::mr::host_accessible>);
static_assert(!cuda::mr::resource_with<HostMemoryResource, cuda::mr::device_accessible>);
}  // namespace rapidsmpf
