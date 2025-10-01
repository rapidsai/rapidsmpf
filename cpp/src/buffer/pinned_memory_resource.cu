/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime_api.h>

#include <cuda/experimental/memory_resource.cuh>

#include <rapidsmpf/buffer/pinned_memory_resource.hpp>
#include <rapidsmpf/error.hpp>

namespace rapidsmpf {

namespace {
cuda::experimental::memory_pool_properties get_memory_pool_properties(
    PinnedPoolProperties const& properties
) {
    return cuda::experimental::memory_pool_properties{
        .initial_pool_size = properties.initial_pool_size,
        .release_threshold = properties.release_threshold
    };
}
}  // namespace

// PinnedMemoryPool implementation
struct PinnedMemoryPool::PinnedMemoryPoolImpl {
    PinnedMemoryPoolImpl(
        int numa_id, cuda::experimental::memory_pool_properties properties
    )
        : numa_id(numa_id), p_pool{numa_id, std::move(properties)} {}

    int numa_id;
    cuda::experimental::pinned_memory_pool p_pool;
};

PinnedMemoryPool::PinnedMemoryPool(int numa_id, PinnedPoolProperties properties)
    : numa_id_(numa_id),
      properties_(std::move(properties)),
      impl_(std::make_unique<PinnedMemoryPoolImpl>(
          numa_id, get_memory_pool_properties(properties_)
      )) {}

PinnedMemoryPool::~PinnedMemoryPool() = default;

// PinnedMemoryResource implementation
struct PinnedMemoryResource::PinnedMemoryResourceImpl {
    PinnedMemoryResourceImpl(cuda::experimental::pinned_memory_pool& pool)
        : p_resource{pool} {}

    void* allocate_async(size_t bytes, const cuda::stream_ref stream_ref) {
        return p_resource.allocate_async(bytes, stream_ref);
    }

    void deallocate_async(void* ptr, const cuda::stream_ref stream_ref) {
        p_resource.deallocate_async(ptr, size_t{}, stream_ref);
    }

    cuda::experimental::pinned_memory_resource p_resource;
};

PinnedMemoryResource::PinnedMemoryResource(PinnedMemoryPool& pool)
    : impl_(std::make_unique<PinnedMemoryResourceImpl>(
          pool.impl_->p_pool
      )) {}

PinnedMemoryResource::~PinnedMemoryResource() = default;

void* PinnedMemoryResource::allocate_async(
    size_t bytes, const cuda::stream_ref stream_ref
) {
    return impl_->allocate_async(bytes, stream_ref);
}

void PinnedMemoryResource::deallocate_async(
    void* ptr, const cuda::stream_ref stream_ref
) {
    impl_->deallocate_async(ptr, stream_ref);
}

// PinnedHostBuffer implementation
PinnedHostBuffer::PinnedHostBuffer(
    size_t size, cuda::stream_ref stream, PinnedMemoryResource* p_resource
)
    : size_(size), stream_ref_(stream), p_resource_(p_resource) {
    RAPIDSMPF_EXPECTS(p_resource_ != nullptr, "p_resource cannot be nullptr");
    data_ = p_resource_->allocate_async(size, stream);
}

PinnedHostBuffer::PinnedHostBuffer(
    void const* data,
    size_t size,
    cuda::stream_ref stream,
    PinnedMemoryResource* p_resource
)
    : PinnedHostBuffer(size, stream, p_resource) {
    if (size > 0) {
        RAPIDSMPF_EXPECTS(nullptr != data, "Invalid copy from nullptr.");
        RAPIDSMPF_EXPECTS(nullptr != data_, "Invalid copy to nullptr.");
        RAPIDSMPF_CUDA_TRY(
            cudaMemcpyAsync(data_, data, size, cudaMemcpyDefault, stream.get())
        );
    }
}

PinnedHostBuffer::~PinnedHostBuffer() noexcept {
    deallocate_async();
    stream_ref_.wait();
}

void PinnedHostBuffer::deallocate_async() noexcept {
    if (p_resource_ && data_) {
        p_resource_->impl_->deallocate_async(data_, stream_ref_);
        data_ = nullptr;
        size_ = 0;
    }
}

}  // namespace rapidsmpf
