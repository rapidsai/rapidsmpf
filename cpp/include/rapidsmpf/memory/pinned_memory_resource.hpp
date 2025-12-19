/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <memory>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cuda/memory_resource>

#include <rmm/aligned.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/host_memory_resource.hpp>
#include <rapidsmpf/utils.hpp>


/// @brief The minimum CUDA version required for PinnedMemoryResource.
#define RAPIDSMPF_PINNED_MEM_RES_MIN_CUDA_VERSION 12060
#define RAPIDSMPF_PINNED_MEM_RES_MIN_CUDA_VERSION_STR "v12.6"

namespace rapidsmpf {

/**
 * @brief Checks if the PinnedMemoryResource is supported for the current CUDA version.
 *
 * RapidsMPF requires CUDA 12.6 or newer to support pinned memory resources.
 */
inline bool is_pinned_memory_resources_supported() {
    static const bool supported = [] {
        // check if the device supports async memory pools
        int cuda_pool_supported{};
        auto attr_result = cudaDeviceGetAttribute(
            &cuda_pool_supported,
            cudaDevAttrMemoryPoolsSupported,
            rmm::get_current_cuda_device().value()
        );
        if (attr_result != cudaSuccess || cuda_pool_supported != 1) {
            return false;
        }

        int cuda_driver_version{};
        auto driver_result = cudaDriverGetVersion(&cuda_driver_version);
        int cuda_runtime_version{};
        auto runtime_result = cudaRuntimeGetVersion(&cuda_runtime_version);
        return driver_result == cudaSuccess && runtime_result == cudaSuccess
               && cuda_driver_version >= RAPIDSMPF_PINNED_MEM_RES_MIN_CUDA_VERSION
               && cuda_runtime_version >= RAPIDSMPF_PINNED_MEM_RES_MIN_CUDA_VERSION;
    }();
    return supported;
}

class PinnedMemoryResource;

/**
 * @brief Memory resource that provides pinned (page-locked) host memory using a pool.
 *
 * This resource allocates and deallocates pinned host memory asynchronously through
 * CUDA streams. It offers higher bandwidth and lower latency for device transfers
 * compared to regular pageable host memory.
 */
class PinnedMemoryResource final : public HostMemoryResource {
  public:
    /// @brief Sentinel value used to disable pinned host memory.
    static constexpr auto Disabled = nullptr;

    /**
     * @brief Construct a pinned (page-locked) host memory resource.
     *
     * The pool has no maximum size. To restrict its growth, use
     * `BufferResource::LimitAvailableMemory` or a similar mechanism.
     *
     * @param numa_id NUMA node from which memory should be allocated. By default,
     * the resource uses the NUMA node of the calling thread.
     *
     * @throws rapidsmpf::cuda_error If pinned host memory pools are not supported by
     * the current CUDA version or if CUDA initialization fails.
     */
    PinnedMemoryResource(int numa_id = get_current_numa_node_id());

    /**
     * @brief Create a pinned memory resource if the system supports pinned memory.
     *
     * @param numa_id The NUMA node to associate with the resource. Defaults to the
     * current NUMA node.
     *
     * @return A shared pointer to a new `PinnedMemoryResource` when supported,
     * otherwise `PinnedMemoryResource::Disabled`.
     *
     * @see PinnedMemoryResource::PinnedMemoryResource
     */
    static std::shared_ptr<PinnedMemoryResource> make_if_available(
        int numa_id = get_current_numa_node_id()
    );

    ~PinnedMemoryResource() override;

    /**
     * @brief Allocates pinned host memory associated with a CUDA stream.
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
        rmm::cuda_stream_view stream,
        std::size_t size,
        std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT
    ) override;

    /**
     * @brief Deallocates pinned host memory associated with a CUDA stream.
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
        std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT
    ) noexcept override;

    /**
     * @brief Compares this resource to another resource.
     *
     * Two resources are considered equal if memory allocated by one may be
     * deallocated by the other.
     *
     * @param other The resource to compare with.
     * @return true because all instances of this base class are considered equal.
     */
    [[nodiscard]] bool is_equal(HostMemoryResource const& other) const noexcept override;

    /**
     * @brief Enables the `cuda::mr::host_accessible` property.
     *
     * This property declares that a `HostMemoryResource` provides host accessible memory.
     */
    friend void get_property(
        PinnedMemoryResource const&, cuda::mr::device_accessible
    ) noexcept {}

  private:
    // using PImpl idiom to hide cudax .cuh headers from rapidsmpf. cudax cuh headers will
    // only be used by the impl in .cu file.
    struct PinnedMemoryResourceImpl;
    std::shared_ptr<PinnedMemoryResourceImpl> impl_;
};

static_assert(cuda::mr::resource<PinnedMemoryResource>);
static_assert(cuda::mr::resource_with<PinnedMemoryResource, cuda::mr::host_accessible>);
static_assert(cuda::mr::resource_with<PinnedMemoryResource, cuda::mr::device_accessible>);

}  // namespace rapidsmpf
