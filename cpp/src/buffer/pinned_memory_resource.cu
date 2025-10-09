/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cstddef>
#include <memory>

#include <cuda_runtime_api.h>

#include <cuda/experimental/memory_resource.cuh>
#include <cuda/memory_resource>

#include <rmm/resource_ref.hpp>

#include <rapidsmpf/buffer/pinned_memory_resource.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/utils.hpp>

namespace rapidsmpf {
#if RAPIDSMPF_CUDA_VERSION_AT_LEAST(RAPIDSMPF_PINNED_MEM_RES_MIN_CUDA_VERSION)
namespace {
cuda::experimental::memory_pool_properties get_memory_pool_properties(
    PinnedPoolProperties const&
) {
    return cuda::experimental::memory_pool_properties{};
}
}  // namespace

// PinnedMemoryPool implementation
struct PinnedMemoryPool::PinnedMemoryPoolImpl {
    PinnedMemoryPoolImpl(int numa_id, PinnedPoolProperties const& properties)
        : numa_id(numa_id), p_pool{numa_id, get_memory_pool_properties(properties)} {}

    int numa_id;
    cuda::experimental::pinned_memory_pool p_pool;
};

// PinnedMemoryResource implementation
struct PinnedMemoryResource::PinnedMemoryResourceImpl {
    PinnedMemoryResourceImpl(PinnedMemoryPool& pool) : p_resource{pool.impl_->p_pool} {}

    void* allocate(rmm::cuda_stream_view stream, size_t bytes) {
        return p_resource.allocate(stream, bytes);
    }

    void* allocate(rmm::cuda_stream_view stream, size_t bytes, size_t alignment) {
        return p_resource.allocate(stream, bytes, alignment);
    }

    void deallocate(rmm::cuda_stream_view stream, void* ptr, size_t bytes) {
        p_resource.deallocate(stream, ptr, bytes);
    }

    void deallocate(
        rmm::cuda_stream_view stream, void* ptr, size_t bytes, size_t alignment
    ) {
        p_resource.deallocate(stream, ptr, bytes, alignment);
    }

    void* allocate_sync(size_t bytes, size_t alignment) {
        return p_resource.allocate_sync(bytes, alignment);
    }

    void deallocate_sync(void* ptr, size_t bytes, size_t alignment) {
        p_resource.deallocate_sync(ptr, bytes, alignment);
    }

    cuda::experimental::pinned_memory_resource p_resource;
};
#else  // CUDA_VERSION < RAPIDSMPF_PINNED_MEM_RES_MIN_CUDA_VERSION
struct PinnedMemoryPool::PinnedMemoryPoolImpl {
    PinnedMemoryPoolImpl(int, PinnedPoolProperties const&) {
        RAPIDSMPF_FAIL(
            "PinnedMemoryPool is not supported for CUDA versions "
            "below " RAPIDSMPF_PINNED_MEM_RES_MIN_CUDA_VERSION_STR
        );
    }
};

struct PinnedMemoryResource::PinnedMemoryResourceImpl {
    PinnedMemoryResourceImpl(PinnedMemoryPool&) {
        RAPIDSMPF_FAIL(
            "PinnedMemoryResource is not supported for CUDA versions "
            "below " RAPIDSMPF_PINNED_MEM_RES_MIN_CUDA_VERSION_STR
        );
    }

    void* allocate(rmm::cuda_stream_view, size_t) {
        RAPIDSMPF_FAIL(
            "PinnedMemoryResource::allocate is not supported for CUDA versions "
            "below " RAPIDSMPF_PINNED_MEM_RES_MIN_CUDA_VERSION_STR
        );
    }

    void* allocate(rmm::cuda_stream_view, size_t, size_t) {
        RAPIDSMPF_FAIL(
            "PinnedMemoryResource::allocate is not supported for CUDA versions "
            "below " RAPIDSMPF_PINNED_MEM_RES_MIN_CUDA_VERSION_STR
        );
    }

    void deallocate(rmm::cuda_stream_view, void*, size_t) {
        RAPIDSMPF_FAIL(
            "PinnedMemoryResource::deallocate is not supported for CUDA versions "
            "below " RAPIDSMPF_PINNED_MEM_RES_MIN_CUDA_VERSION_STR
        );
    }

    void deallocate(rmm::cuda_stream_view, void*, size_t, size_t) {
        RAPIDSMPF_FAIL(
            "PinnedMemoryResource::deallocate is not supported for CUDA versions "
            "below " RAPIDSMPF_PINNED_MEM_RES_MIN_CUDA_VERSION_STR
        );
    }

    void* allocate_sync(size_t, size_t) {
        RAPIDSMPF_FAIL(
            "PinnedMemoryResource::allocate_sync is not supported for CUDA versions "
            "below " RAPIDSMPF_PINNED_MEM_RES_MIN_CUDA_VERSION_STR
        );
    }

    void deallocate_sync(void*, size_t, size_t) {
        RAPIDSMPF_FAIL(
            "PinnedMemoryResource::deallocate_sync is not supported for CUDA versions "
            "below " RAPIDSMPF_PINNED_MEM_RES_MIN_CUDA_VERSION_STR
        );
    }
};
#endif

PinnedMemoryPool::PinnedMemoryPool(int numa_id, PinnedPoolProperties properties)
    : numa_id_(numa_id),
      properties_(std::move(properties)),
      impl_(std::make_unique<PinnedMemoryPoolImpl>(numa_id, properties_)) {}

PinnedMemoryPool::~PinnedMemoryPool() = default;

PinnedMemoryResource::PinnedMemoryResource(PinnedMemoryPool& pool)
    : impl_(std::make_unique<PinnedMemoryResourceImpl>(pool)) {}

PinnedMemoryResource::~PinnedMemoryResource() = default;

void* PinnedMemoryResource::allocate(rmm::cuda_stream_view stream, size_t bytes) {
    return impl_->allocate(stream, bytes);
}

void* PinnedMemoryResource::allocate(
    rmm::cuda_stream_view stream, size_t bytes, size_t alignment
) {
    return impl_->allocate(stream, bytes, alignment);
}

void PinnedMemoryResource::deallocate(
    rmm::cuda_stream_view stream, void* ptr, size_t bytes
) noexcept {
    impl_->deallocate(stream, ptr, bytes);
}

void PinnedMemoryResource::deallocate(
    rmm::cuda_stream_view stream, void* ptr, size_t bytes, size_t alignment
) noexcept {
    impl_->deallocate(stream, ptr, bytes, alignment);
}

void* PinnedMemoryResource::allocate_sync(size_t bytes, size_t alignment) {
    return impl_->allocate_sync(bytes, alignment);
}

void PinnedMemoryResource::deallocate_sync(void* ptr, size_t bytes, size_t alignment) {
    impl_->deallocate_sync(ptr, bytes, alignment);
}

// PinnedHostBuffer implementation
PinnedHostBuffer::PinnedHostBuffer(
    size_t size, rmm::cuda_stream_view stream, std::shared_ptr<PinnedMemoryResource> mr
)
    : size_(size), stream_(stream), mr_(std::move(mr)) {
    RAPIDSMPF_EXPECTS(mr_ != nullptr, "mr cannot be nullptr", std::invalid_argument);
    data_ = static_cast<std::byte*>(mr_->allocate(stream, size));
}

PinnedHostBuffer::PinnedHostBuffer(
    void const* src_data,
    size_t size,
    rmm::cuda_stream_view stream,
    std::shared_ptr<PinnedMemoryResource> mr
)
    : PinnedHostBuffer(size, stream, std::move(mr)) {
    if (size > 0) {
        RAPIDSMPF_EXPECTS(
            nullptr != src_data, "Invalid copy from nullptr.", std::invalid_argument
        );
        RAPIDSMPF_EXPECTS(
            nullptr != data_, "Invalid copy to nullptr.", std::invalid_argument
        );
        RAPIDSMPF_CUDA_TRY(
            cudaMemcpyAsync(data_, src_data, size, cudaMemcpyDefault, stream.value())
        );
    }
}

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

PinnedHostBuffer& PinnedHostBuffer::operator=(PinnedHostBuffer&& other) {
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
        mr_->deallocate(stream_, data_, size_);
        data_ = nullptr;
        size_ = 0;
    }
}

void PinnedHostBuffer::synchronize() {
    stream_.synchronize();
}

}  // namespace rapidsmpf
