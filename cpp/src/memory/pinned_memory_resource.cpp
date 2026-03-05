/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <limits>

#include <cuda_runtime_api.h>

#include <cuda/memory_resource>

#include <rmm/mr/pinned_host_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/pinned_memory_resource.hpp>
#include <rapidsmpf/utils/misc.hpp>

namespace rapidsmpf {

namespace {
cuda::memory_pool_properties get_memory_pool_properties() {
    return cuda::memory_pool_properties{
        // It was observed that priming async pools have little effect for performance.
        // See <https://github.com/rapidsai/rmm/issues/1931>.
        .initial_pool_size = 0,
        // Before <https://github.com/NVIDIA/cccl/pull/6718>, the default
        // `release_threshold` was 0, which defeats the purpose of having a pool. We
        // now set it so the pool never releases unused pinned memory.
        .release_threshold = std::numeric_limits<std::size_t>::max(),
        // This defines how the allocations can be exported (IPC). See the docs of
        // `cudaMemPoolCreate` in <https://docs.nvidia.com/cuda/cuda-runtime-api>.
        .allocation_handle_type = ::cudaMemAllocationHandleType::cudaMemHandleTypeNone
    };
}

std::shared_ptr<cuda::pinned_memory_pool> make_pinned_memory_pool(
    int numa_id, cuda::memory_pool_properties const& props
) {
    RAPIDSMPF_EXPECTS(
        is_pinned_memory_resources_supported(),
        "Pinned host memory is not supported on this system. "
        "CUDA " RAPIDSMPF_PINNED_MEM_RES_MIN_CUDA_VERSION_STR
        " is one of the requirements, but additional platform or driver constraints may "
        "apply. If needed, use `PinnedMemoryResource::Disabled` to disable pinned host "
        "memory, noting that this may significantly degrade spilling performance.",
        std::invalid_argument
    );
    return std::make_shared<cuda::pinned_memory_pool>(numa_id, props);
}
}  // namespace

PinnedMemoryResource::PinnedMemoryResource(int numa_id)
    : pool_{make_pinned_memory_pool(numa_id, get_memory_pool_properties())} {}

std::shared_ptr<PinnedMemoryResource> PinnedMemoryResource::make_if_available(
    int numa_id
) {
    if (is_pinned_memory_resources_supported()) {
        return std::make_shared<rapidsmpf::PinnedMemoryResource>(numa_id);
    }
    return PinnedMemoryResource::Disabled;
}

std::shared_ptr<PinnedMemoryResource> PinnedMemoryResource::from_options(
    config::Options options
) {
    bool const pinned_memory = options.get<bool>("pinned_memory", [](auto const& s) {
        return parse_string<bool>(s.empty() ? "False" : s);
    });

    return pinned_memory ? PinnedMemoryResource::make_if_available()
                         : PinnedMemoryResource::Disabled;
}

PinnedMemoryResource::~PinnedMemoryResource() = default;

void* PinnedMemoryResource::allocate(
    rmm::cuda_stream_view stream, std::size_t bytes, std::size_t alignment
) {
    return pool_->allocate(stream, bytes, alignment);
}

void PinnedMemoryResource::deallocate(
    rmm::cuda_stream_view stream, void* ptr, std::size_t bytes, std::size_t alignment
) noexcept {
    pool_->deallocate(stream, ptr, bytes, alignment);
}

void* PinnedMemoryResource::allocate_sync(std::size_t bytes, std::size_t alignment) {
    return pool_->allocate_sync(bytes, alignment);
}

void PinnedMemoryResource::deallocate_sync(
    void* ptr, std::size_t bytes, std::size_t alignment
) {
    pool_->deallocate_sync(ptr, bytes, alignment);
}

std::shared_ptr<PinnedMemoryResource> PinnedMemoryResource::make_fixed_sized_if_available(
    int numa_id,
    std::size_t mem_limit,
    std::size_t capacity,
    std::size_t block_size,
    std::size_t pool_size,
    std::size_t initial_pools
) {
    if (!is_pinned_memory_resources_supported()) {
        return PinnedMemoryResource::Disabled;
    }
    auto mr = std::make_shared<PinnedMemoryResource>(numa_id);
    mr->fixed_size_host_mr_ =
        std::make_shared<FixedSizedHostMemoryResource>(
            rmm::get_current_cuda_device().value(),
            *mr,
            mem_limit,
            capacity,
            block_size,
            pool_size,
            initial_pools
        );
    return mr;
}

PinnedMemoryResource::FixedSizedBlocksAllocation PinnedMemoryResource::allocate_fixed_sized(
    std::size_t size
) {
    RAPIDSMPF_EXPECTS(
        fixed_size_host_mr_ != nullptr,
        "fixed-size host memory resource not initialized; "
        "use make_fixed_sized_if_available to create this resource",
        std::invalid_argument
    );
    return fixed_size_host_mr_->allocate_multiple_blocks(size);
}

bool PinnedMemoryResource::is_equal(HostMemoryResource const& other) const noexcept {
    auto const* o = dynamic_cast<PinnedMemoryResource const*>(&other);
    return o != nullptr && pool_ == o->pool_;
}

}  // namespace rapidsmpf
