/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cuda/memory_resource>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/host_buffer.hpp>
#include <rapidsmpf/utils.hpp>


/// @brief The minimum CUDA version required for PinnedMemoryResource.
#define RAPIDSMPF_PINNED_MEM_RES_MIN_CUDA_VERSION 12060
#define RAPIDSMPF_PINNED_MEM_RES_MIN_CUDA_VERSION_STR \
    RAPIDSMPF_STRINGIFY(RAPIDSMPF_PINNED_MEM_RES_MIN_CUDA_VERSION)

namespace rapidsmpf {

/**
 * @brief Checks if the PinnedMemoryResource is supported for the current CUDA version.
 *
 * Requires rapidsmpf to be build with cuda>=12.6.
 *
 * @note The driver version check is cached and only performed once.
 */
inline bool is_pinned_memory_resources_supported() {
#if RAPIDSMPF_CUDA_VERSION_AT_LEAST(RAPIDSMPF_PINNED_MEM_RES_MIN_CUDA_VERSION)
    static const bool supported = [] {
        int driver_version = 0;
        RAPIDSMPF_CUDA_TRY(cudaDriverGetVersion(&driver_version));
        return driver_version >= RAPIDSMPF_PINNED_MEM_RES_MIN_CUDA_VERSION;
    }();
    return supported;
#else
    return false;
#endif
}

class PinnedMemoryResource;  // forward declaration

/**
 * @brief Properties for configuring a pinned memory pool. It is aimed to mimic
 * `cuda::experimental::memory_pool_properties`.
 *
 * @sa
 * https://nvidia.github.io/cccl/cudax/api/structcuda_1_1experimental_1_1memory__pool__properties.html
 *
 * Currently, this is a placeholder and does not have any effect. It was observed that
 * priming async pools have little effect for performance.
 *
 * @sa https://github.com/rapidsai/rmm/issues/1931
 */
struct PinnedPoolProperties {};

/**
 * @brief A pinned host memory pool for stream-ordered allocations/deallocations. This
 * internally uses `cuda::experimental::pinned_memory_pool`.
 *
 * @sa
 * https://nvidia.github.io/cccl/cudax/api/classcuda_1_1experimental_1_1pinned__memory__pool.html
 */
class PinnedMemoryPool {
    friend class PinnedMemoryResource;

  public:
    /**
     * @brief Constructs a new pinned memory pool.
     *
     * @param properties Configuration properties for the memory pool.
     *
     * @throws rapidsmpf::cuda_error If the pinned memory pool is not supported for the
     * current CUDA version.
     */
    PinnedMemoryPool(PinnedPoolProperties properties = {});

    /**
     * @brief Constructs a new pinned memory pool.
     *
     * @param numa_id The optional NUMA node ID to associate with this pool.
     * @param properties Configuration properties for the memory pool.
     *
     * @throws rapidsmpf::cuda_error If the pinned memory pool is not supported for the
     * current CUDA version.
     */
    PinnedMemoryPool(int numa_id, PinnedPoolProperties properties = {});

    /**
     * @brief Destroys the pinned memory pool.
     *
     * Releases all memory associated with this pool.
     */
    ~PinnedMemoryPool();

    /**
     * @brief Gets the properties used to configure this pool.
     *
     * @return A const reference to the pool properties.
     */
    [[nodiscard]] constexpr PinnedPoolProperties const& properties() const noexcept {
        return properties_;
    }

  private:
    PinnedPoolProperties properties_;  ///< Configuration properties for this pool.

    // using PImpl idiom to hide cudax .cuh headers from rapidsmpf. cudax cuh headers will
    // only be used by the impl in .cu file.
    struct PinnedMemoryPoolImpl;
    std::unique_ptr<PinnedMemoryPoolImpl> impl_;
};

/// @brief The default alignment for pinned memory allocations.
constexpr size_t default_pinned_memory_alignment =
    cuda::mr::default_cuda_malloc_alignment;

/**
 * @brief A memory resource that allocates/deallocates pinned host memory from a pinned
 * host memory pool. This internally uses
 * `cuda::experimental::pinned_memory_resource`.
 *
 * @sa
 * https://nvidia.github.io/cccl/cudax/api/classcuda_1_1experimental_1_1pinned__memory__resource.html
 *
 * This class provides an interface for allocating and deallocating pinned
 * (page-locked) host memory asynchronously using CUDA streams.
 */
class PinnedMemoryResource {
  public:
    /**
     * @brief Friend function to get the host_accessible property.
     */
    friend constexpr void get_property(
        const PinnedMemoryResource&, cuda::mr::host_accessible
    ) noexcept {}

    /**
     * @brief Constructs a new pinned memory resource.
     *
     * @param pool The pinned memory pool to use for allocations.
     *
     * @throws rapidsmpf::cuda_error If the pinned memory resource is not supported for
     * the current CUDA version.
     */
    PinnedMemoryResource(PinnedMemoryPool& pool);

    /**
     * @brief Destroys the pinned memory resource.
     *
     * Note: This does not deallocate memory that was allocated through this resource.
     * All allocated memory should be explicitly deallocated before destruction.
     */
    ~PinnedMemoryResource();

    /**
     * @brief Allocates pinned memory asynchronously.
     *
     * @param stream The CUDA stream to use for the allocation operation.
     * @param bytes The number of bytes to allocate.
     * @return A pointer to the allocated memory, or nullptr if allocation failed.
     */
    void* allocate(rmm::cuda_stream_view stream, size_t bytes);

    /**
     * @brief Allocates pinned memory asynchronously with alignment.
     *
     * @param stream The CUDA stream to use for the allocation operation.
     * @param bytes The number of bytes to allocate.
     * @param alignment The alignment requirement for the allocation.
     * @return A pointer to the allocated memory, or nullptr if allocation failed.
     */
    void* allocate(rmm::cuda_stream_view stream, size_t bytes, size_t alignment);

    /**
     * @brief Allocates pinned memory synchronously.
     *
     * @param bytes The number of bytes to allocate.
     * @param alignment The alignment requirement for the allocation.
     * @return A pointer to the allocated memory, or nullptr if allocation failed.
     */
    void* allocate_sync(size_t bytes, size_t alignment = default_pinned_memory_alignment);

    /**
     * @brief Deallocates pinned memory asynchronously.
     *
     * @param stream The CUDA stream to use for the deallocation operation.
     * @param ptr A pointer to the memory to deallocate.
     * @param bytes The size of the memory to deallocate.
     */
    void deallocate(rmm::cuda_stream_view stream, void* ptr, size_t bytes) noexcept;

    /**
     * @brief Deallocates pinned memory asynchronously with alignment.
     *
     * @param stream The CUDA stream to use for the deallocation operation.
     * @param ptr A pointer to the memory to deallocate.
     * @param bytes The size of the memory to deallocate.
     * @param alignment The alignment that was used for allocation.
     */
    void deallocate(
        rmm::cuda_stream_view stream, void* ptr, size_t bytes, size_t alignment
    ) noexcept;

    /**
     * @brief Deallocates pinned memory synchronously with alignment.
     *
     * @param ptr A pointer to the memory to deallocate.
     * @param bytes The size of the memory to deallocate.
     * @param alignment The alignment that was used for allocation.
     */
    void deallocate_sync(
        void* ptr, size_t bytes, size_t alignment = default_pinned_memory_alignment
    );

    /**
     * @brief equality operator
     *
     * @param other The other pinned memory resource to compare with.
     * @return True if the two pinned memory resources are equal, false otherwise.
     */
    bool operator==(const PinnedMemoryResource& other) const noexcept {
        return impl_ == other.impl_;
    }

  private:
    // using PImpl idiom to hide cudax .cuh headers from rapidsmpf. cudax cuh headers will
    // only be used by the impl in .cu file.
    struct PinnedMemoryResourceImpl;
    std::unique_ptr<PinnedMemoryResourceImpl> impl_;
};

static_assert(cuda::mr::resource_with<PinnedMemoryResource, cuda::mr::host_accessible>);

}  // namespace rapidsmpf
