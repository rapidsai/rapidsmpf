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
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/host_memory_resource.hpp>
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

/**
 * @brief Memory resource that provides pinned (page-locked) host memory using a pool.
 *
 * This resource allocates and deallocates pinned host memory asynchronously through
 * CUDA streams. It offers higher bandwidth and lower latency for device transfers
 * compared to regular pageable host memory.
 */
class PinnedMemoryResource final : public HostMemoryResource {
  public:
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

  private:
    // using PImpl idiom to hide cudax .cuh headers from rapidsmpf. cudax cuh headers will
    // only be used by the impl in .cu file.
    struct PinnedMemoryResourceImpl;
    std::unique_ptr<PinnedMemoryResourceImpl> impl_;
};

}  // namespace rapidsmpf
