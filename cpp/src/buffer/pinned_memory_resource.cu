/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>

#include <cuda_runtime_api.h>

#include <cuda/experimental/memory_resource.cuh>

#include <rapidsmpf/buffer/pinned_memory_resource.hpp>
#include <rapidsmpf/error.hpp>

namespace rapidsmpf {

namespace {
cuda::experimental::memory_pool_properties get_memory_pool_properties(
    PinnedPoolProperties const&
) {
    return cuda::experimental::memory_pool_properties{};
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
      impl_(
          std::make_unique<PinnedMemoryPoolImpl>(
              numa_id, get_memory_pool_properties(properties_)
          )
      ) {}

PinnedMemoryPool::~PinnedMemoryPool() = default;

// PinnedMemoryResource implementation
struct PinnedMemoryResource::PinnedMemoryResourceImpl {
    PinnedMemoryResourceImpl(cuda::experimental::pinned_memory_pool& pool)
        : p_resource{pool} {}

    void* allocate_async(size_t bytes, rmm::cuda_stream_view stream) {
        return p_resource.allocate(stream, bytes);
    }

    void deallocate_async(void* ptr, rmm::cuda_stream_view stream) {
        p_resource.deallocate(stream, ptr, size_t{});
    }

    cuda::experimental::pinned_memory_resource p_resource;
};

PinnedMemoryResource::PinnedMemoryResource(PinnedMemoryPool& pool)
    : impl_(std::make_unique<PinnedMemoryResourceImpl>(pool.impl_->p_pool)) {}

PinnedMemoryResource::~PinnedMemoryResource() = default;

void* PinnedMemoryResource::allocate_async(size_t bytes, rmm::cuda_stream_view stream) {
    return impl_->allocate_async(bytes, stream);
}

void PinnedMemoryResource::deallocate_async(void* ptr, rmm::cuda_stream_view stream) {
    impl_->deallocate_async(ptr, stream);
}

// PinnedHostBuffer implementation
PinnedHostBuffer::PinnedHostBuffer(
    size_t size, rmm::cuda_stream_view stream, std::shared_ptr<PinnedMemoryResource> mr
)
    : size_(size), stream_(stream), mr_(std::move(mr)) {
    RAPIDSMPF_EXPECTS(mr_ != nullptr, "mr cannot be nullptr");
    data_ = static_cast<std::byte*>(mr_->allocate_async(size, stream));
}

PinnedHostBuffer::PinnedHostBuffer(
    void const* src_data,
    size_t size,
    rmm::cuda_stream_view stream,
    std::shared_ptr<PinnedMemoryResource> mr
)
    : PinnedHostBuffer(size, stream, std::move(mr)) {
    if (size > 0) {
        RAPIDSMPF_EXPECTS(nullptr != src_data, "Invalid copy from nullptr.");
        RAPIDSMPF_EXPECTS(nullptr != data_, "Invalid copy to nullptr.");
        RAPIDSMPF_CUDA_TRY(
            cudaMemcpyAsync(data_, src_data, size, cudaMemcpyDefault, stream.value())
        );
    }
}

PinnedHostBuffer::PinnedHostBuffer(
    PinnedHostBuffer const& other,
    rmm::cuda_stream_view stream,
    std::shared_ptr<PinnedMemoryResource> mr
)
    : PinnedHostBuffer(other.data_, other.size_, stream, std::move(mr)) {}

PinnedHostBuffer::~PinnedHostBuffer() noexcept {
    deallocate_async();
}

PinnedHostBuffer::PinnedHostBuffer(PinnedHostBuffer&& other)
    : data_(other.data_),
      size_(other.size_),
      stream_(other.stream_),
      mr_(std::move(other.mr_)) {
    other.data_ = nullptr;
    other.size_ = 0;
}

PinnedHostBuffer& PinnedHostBuffer::operator=(PinnedHostBuffer&& other) noexcept {
    if (this != &other) {
        deallocate_async();
        data_ = std::exchange(other.data_, nullptr);
        size_ = other.size_;
        stream_ = other.stream_;
        mr_ = std::move(other.mr_);
    }
    return *this;
}

void PinnedHostBuffer::deallocate_async() noexcept {
    if (mr_ && data_) {
        mr_->deallocate_async(data_, stream_);
        data_ = nullptr;
        size_ = 0;
    }
}

void PinnedHostBuffer::synchronize() {
    stream_.synchronize();
}

}  // namespace rapidsmpf
