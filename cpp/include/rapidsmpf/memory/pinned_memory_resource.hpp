/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <memory>
#include <optional>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cuda/memory_resource>

#include <rmm/aligned.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <rapidsmpf/config.hpp>
#include <rapidsmpf/detail/rmm_resource_adaptor_impl.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/back_ref_mixin.hpp>
#include <rapidsmpf/system_info.hpp>
#include <rapidsmpf/utils/misc.hpp>

/// @brief The minimum CUDA version required for PinnedMemoryResource.
// NOLINTBEGIN(modernize-macro-to-enum)
#define RAPIDSMPF_PINNED_MEM_RES_MIN_CUDA_VERSION 12060
#define RAPIDSMPF_PINNED_MEM_RES_MIN_CUDA_VERSION_STR "v12.6"

// NOLINTEND(modernize-macro-to-enum)

namespace rapidsmpf {

class BufferResource;

/**
 * @brief Checks if the PinnedMemoryResource is supported for the current CUDA version.
 *
 * RapidsMPF requires CUDA 12.6 or newer to support pinned memory resources.
 *
 * @return True if the PinnedMemoryResource is supported for the current CUDA version,
 * false otherwise.
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

/**
 * @brief Sentinel used to disable pinned host memory.
 *
 * Pass this in place of a `PinnedPoolProperties` (e.g. to `BufferResource::create()`)
 * to disable pinned host memory allocations.
 */
constexpr std::nullopt_t PinnedMemoryDisabled = std::nullopt;

/**
 * @brief Properties for configuring a pinned memory pool.
 */
struct PinnedPoolProperties {
    /// @brief Initial size of the pool. Initial size is important for pinned memory
    /// performance, especially for the first allocation. (See
    /// `BM_PinnedFirstAlloc_InitialPoolSize` benchmark.)
    std::size_t initial_pool_size = 0;

    /// @brief Maximum size of the pool. `std::nullopt` means no limit.
    std::optional<std::size_t> max_pool_size = std::nullopt;

    /// @brief NUMA node from which pinned memory should be allocated. Defaults to
    /// the NUMA node of the calling thread.
    int numa_id = get_current_numa_node();
};

/**
 * @brief Parse pinned memory pool properties from configuration options.
 *
 * Recognized options:
 * - "pinned_memory": enable pinned memory.
 * - "pinned_initial_pool_size" (bytes or percentage): initial pool size.
 *   - Byte values (e.g. "1 MiB") are applied literally.
 *   - Percentages (e.g. "10%") are relative to `get_host_memory_per_gpu()`.
 * - "pinned_max_pool_size" (bytes, percentage, or disabled): maximum pool size.
 *   - Byte and percentages uses the same parsing rules as "pinned_initial_pool_size".
 *   - A disabled value (e.g. "off") leaves the pool unbounded.
 *
 * @param options Configuration options.
 * @return The parsed `PinnedPoolProperties` when "pinned_memory" is enabled,
 * otherwise `std::nullopt` (pinned host memory disabled).
 */
std::optional<PinnedPoolProperties> pinned_pool_properties_from_options(
    config::Options options
);

/**
 * @brief Memory resource that provides pinned (page-locked) host memory using a pool.
 *
 * Inherits from
 * `cuda::mr::shared_resource<RmmResourceAdaptorImpl<cuda::pinned_memory_pool>>`, which
 * holds the pool directly inside the shared control block — no extra heap allocation for
 * the pool itself. Copies share the same underlying pool and memory statistics.
 *
 * This resource allocates and deallocates pinned host memory asynchronously through
 * CUDA streams. It offers higher bandwidth and lower latency for device transfers
 * compared to regular pageable host memory.
 */
class PinnedMemoryResource final
    : public cuda::mr::shared_resource<
          detail::RmmResourceAdaptorImpl<cuda::pinned_memory_pool>>,
      public BackRefMixin<BufferResource> {
    using shared_base = cuda::mr::shared_resource<
        detail::RmmResourceAdaptorImpl<cuda::pinned_memory_pool>>;

  public:
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

    [[nodiscard]] void* allocate(
        cuda::stream_ref stream,
        std::size_t size,
        std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT
    ) {
        return get().allocate(stream, size, alignment);
    }

    /**
     * @brief Deallocates pinned host memory associated with a CUDA stream.
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
    ) noexcept {
        get().deallocate(stream, ptr, size, alignment);
    }

    /**
     * @brief Equality comparison.
     *
     * @param other The other resource to compare.
     * @return True if the two resources share the same underlying shared state.
     */
    [[nodiscard]] bool operator==(PinnedMemoryResource const& other) const noexcept {
        return get() == other.get();
    }

    /**
     * @brief Returns the total number of currently allocated bytes.
     *
     * @return The total number of currently allocated bytes.
     */
    [[nodiscard]] std::int64_t current_allocated() const noexcept {
        return get().current_allocated();
    }

    /**
     * @brief Returns the main memory record for the pinned pool.
     *
     * @return The main memory record for the pinned pool.
     */
    [[nodiscard]] ScopedMemoryRecord get_main_memory_record() const {
        return get().get_main_record();
    }

    /**
     * @brief Returns the properties used to configure the pool.
     *
     * @return The properties used to configure the pool.
     */
    [[nodiscard]] constexpr PinnedPoolProperties const& properties() const noexcept {
        return pool_properties_;
    }

    /**
     * @brief Enables the `cuda::mr::host_accessible` property.
     */
    friend void get_property(
        PinnedMemoryResource const&, cuda::mr::host_accessible
    ) noexcept {}

  private:
    /**
     * @brief Construct a pinned (page-locked) host memory resource.
     *
     * Private: use `BufferResource` to construct instances.
     *
     * @param pool_properties Properties for configuring the pinned memory pool,
     * including the NUMA node from which memory should be allocated.
     *
     * @throws std::invalid_argument If pinned host memory pools are not supported.
     */
    explicit PinnedMemoryResource(PinnedPoolProperties pool_properties);

    friend class BufferResource;

    PinnedPoolProperties pool_properties_;  ///< properties used to configure the pool
};

static_assert(cuda::mr::resource<PinnedMemoryResource>);
static_assert(cuda::mr::resource_with<PinnedMemoryResource, cuda::mr::host_accessible>);
static_assert(cuda::mr::resource_with<PinnedMemoryResource, cuda::mr::device_accessible>);

}  // namespace rapidsmpf
