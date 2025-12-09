/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cstddef>
#include <memory>

#include <cuda_runtime_api.h>

#include <cuda/memory_resource>

#include <rmm/resource_ref.hpp>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/pinned_memory_resource.hpp>
#include <rapidsmpf/utils.hpp>

#if RAPIDSMPF_CUDA_VERSION_AT_LEAST(RAPIDSMPF_PINNED_MEM_RES_MIN_CUDA_VERSION)
#include <cuda/experimental/memory_resource.cuh>
#endif

namespace rapidsmpf {

#if RAPIDSMPF_CUDA_VERSION_AT_LEAST(RAPIDSMPF_PINNED_MEM_RES_MIN_CUDA_VERSION)

namespace {
cuda::experimental::memory_pool_properties get_memory_pool_properties(
    PinnedPoolProperties const&
) {
    return cuda::experimental::memory_pool_properties{};
}
}  // namespace

struct PinnedMemoryResource::PinnedMemoryResourceImpl {
    PinnedMemoryResourceImpl(PinnedPoolProperties const& properties, int numa_id)

        : pool{numa_id, get_memory_pool_properties(properties)}, resource{pool} {}

    void* allocate(rmm::cuda_stream_view stream, size_t bytes, size_t alignment) {
        return resource.allocate(stream, bytes, alignment);
    }

    void deallocate(
        rmm::cuda_stream_view stream, void* ptr, size_t bytes, size_t alignment
    ) {
        resource.deallocate(stream, ptr, bytes, alignment);
    }

    cuda::experimental::pinned_memory_pool pool;
    cuda::experimental::pinned_memory_resource resource;
};
#else  // CUDA_VERSION < RAPIDSMPF_PINNED_MEM_RES_MIN_CUDA_VERSION

struct PinnedMemoryResource::PinnedMemoryResourceImpl {
    PinnedMemoryResourceImpl(PinnedPoolProperties const&, int) {
        RAPIDSMPF_FAIL(
            "PinnedMemoryResource is not supported for CUDA versions "
            "below " RAPIDSMPF_PINNED_MEM_RES_MIN_CUDA_VERSION_STR
        );
    }

    void* allocate(rmm::cuda_stream_view, size_t, size_t) {
        return nullptr;
    }

    void deallocate(rmm::cuda_stream_view, void*, size_t, size_t) {}
};
#endif

PinnedMemoryResource::PinnedMemoryResource(PinnedPoolProperties properties, int numa_id)
    : impl_(std::make_unique<PinnedMemoryResourceImpl>(properties, numa_id)) {
    RAPIDSMPF_EXPECTS(
        is_pinned_memory_resources_supported(),
        "PinnedMemoryResource is not supported for CUDA versions "
        "below " RAPIDSMPF_PINNED_MEM_RES_MIN_CUDA_VERSION_STR
    );
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
