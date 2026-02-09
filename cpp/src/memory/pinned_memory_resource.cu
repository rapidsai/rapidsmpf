/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cstddef>
#include <memory>

#include <cuda_runtime_api.h>

#include <rmm/resource_ref.hpp>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/pinned_memory_resource.hpp>
#include <rapidsmpf/utils/misc.hpp>

#if RAPIDSMPF_CUDA_VERSION_AT_LEAST(RAPIDSMPF_PINNED_MEM_RES_MIN_CUDA_VERSION)
#if CCCL_MAJOR_VERSION > 3 || (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION >= 2)
#include <cuda/memory_resource>
#else
#include <cuda/experimental/memory_resource.cuh>
#endif
#endif

namespace rapidsmpf {

#if RAPIDSMPF_CUDA_VERSION_AT_LEAST(RAPIDSMPF_PINNED_MEM_RES_MIN_CUDA_VERSION)

namespace {
#if CCCL_MAJOR_VERSION > 3 || (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION >= 2)
cuda::memory_pool_properties get_memory_pool_properties(
    PinnedPoolProperties const& props
) {
    return cuda::memory_pool_properties{
        // It was observed that priming async pools have little effect for performance.
        // See <https://github.com/rapidsai/rmm/issues/1931>.
        .initial_pool_size = props.initial_pool_size,
        // Before <https://github.com/NVIDIA/cccl/pull/6718>, the default
        // `release_threshold` was 0, which defeats the purpose of having a pool. We
        // now set it so the pool never releases unused pinned memory.
        .release_threshold = props.max_pool_size > 0 ? props.max_pool_size
                                                     : std::numeric_limits<size_t>::max(),
        // This defines how the allocations can be exported (IPC). See the docs of
        // `cudaMemPoolCreate` in <https://docs.nvidia.com/cuda/cuda-runtime-api>.
        .allocation_handle_type = ::cudaMemAllocationHandleType::cudaMemHandleTypeNone,
        .max_pool_size = props.max_pool_size
    };
}
#else
cuda::experimental::memory_pool_properties get_memory_pool_properties(
    PinnedPoolProperties const& props
) {
    return cuda::experimental::memory_pool_properties{
        // It was observed that priming async pools have little effect for performance.
        // See <https://github.com/rapidsai/rmm/issues/1931>.
        .initial_pool_size = props.initial_pool_size,
        // Before <https://github.com/NVIDIA/cccl/pull/6718>, the default
        // `release_threshold` was 0, which defeats the purpose of having a pool. We
        // now set it so the pool never releases unused pinned memory.
        .release_threshold = props.max_pool_size > 0 ? props.max_pool_size
                                                     : std::numeric_limits<size_t>::max(),
        // This defines how the allocations can be exported (IPC). See the docs of
        // `cudaMemPoolCreate` in <https://docs.nvidia.com/cuda/cuda-runtime-api>.
        .allocation_handle_type =
            cuda::experimental::cudaMemAllocationHandleType::cudaMemHandleTypeNone
    };
}
#endif
}  // namespace

struct PinnedMemoryResource::PinnedMemoryResourceImpl {
    PinnedMemoryResourceImpl(int numa_id, PinnedPoolProperties const& props)
        : pool{numa_id, get_memory_pool_properties(props)}, resource{pool} {}

    void* allocate(rmm::cuda_stream_view stream, size_t bytes, size_t alignment) {
        return resource.allocate(stream, bytes, alignment);
    }

    void deallocate(
        rmm::cuda_stream_view stream, void* ptr, size_t bytes, size_t alignment
    ) {
        resource.deallocate(stream, ptr, bytes, alignment);
    }

#if CCCL_MAJOR_VERSION > 3 || (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION >= 2)
    cuda::pinned_memory_pool pool;
    cuda::pinned_memory_pool_ref resource;
#else
    cuda::experimental::pinned_memory_pool pool;
    cuda::experimental::pinned_memory_resource resource;
#endif
};
#else  // CUDA_VERSION < RAPIDSMPF_PINNED_MEM_RES_MIN_CUDA_VERSION

struct PinnedMemoryResource::PinnedMemoryResourceImpl {
    PinnedMemoryResourceImpl(int, PinnedPoolProperties const&) {}

    void* allocate(rmm::cuda_stream_view, size_t, size_t) {
        return nullptr;
    }

    void deallocate(rmm::cuda_stream_view, void*, size_t, size_t) {}
};
#endif

PinnedMemoryResource::PinnedMemoryResource(
    int numa_id, PinnedPoolProperties pool_properties
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
    impl_ = std::make_unique<PinnedMemoryResourceImpl>(numa_id, pool_properties);
}

std::shared_ptr<PinnedMemoryResource> PinnedMemoryResource::make_if_available(
    int numa_id, PinnedPoolProperties pool_properties
) {
    if (is_pinned_memory_resources_supported()) {
        return std::make_shared<rapidsmpf::PinnedMemoryResource>(
            numa_id, pool_properties
        );
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
    rmm::cuda_stream_view stream, size_t bytes, size_t alignment
) {
    return impl_->allocate(stream, bytes, alignment);
}

void PinnedMemoryResource::deallocate(
    rmm::cuda_stream_view stream, void* ptr, size_t bytes, size_t alignment
) noexcept {
    impl_->deallocate(stream, ptr, bytes, alignment);
}

bool PinnedMemoryResource::is_equal(HostMemoryResource const& other) const noexcept {
    auto cast = dynamic_cast<PinnedMemoryResource const*>(&other);
    return cast != nullptr && impl_ == cast->impl_;
}

}  // namespace rapidsmpf
