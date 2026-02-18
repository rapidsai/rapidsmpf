/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */


#include <algorithm>
#include <ranges>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/fixed_sized_host_buffer.hpp>
#include <rapidsmpf/owning_wrapper.hpp>

#include <cucascade/memory/fixed_size_host_memory_resource.hpp>

namespace rapidsmpf {
namespace {

template <typename T>
struct VectorStorage {
    std::vector<std::byte*> block_ptrs;
    T storage;

    static void delete_storage(void* v) {
        delete static_cast<VectorStorage<T>*>(v);
    }
};


}  // namespace

FixedSizedHostBuffer FixedSizedHostBuffer::from_vector(
    std::vector<std::byte> vec, std::size_t block_size
) {
    if (vec.empty()) {
        return FixedSizedHostBuffer(
            std::size_t(0), block_size, std::span<std::byte*>{}, OwningWrapper()
        );
    }

    std::size_t total_size = vec.size();
    auto storage = new VectorStorage<std::vector<std::byte>>();
    storage->block_ptrs.reserve((total_size + block_size - 1) / block_size);
    for (std::size_t i = 0; i < total_size; i += block_size) {
        storage->block_ptrs.push_back(vec.data() + i);
    }
    storage->storage = std::move(vec);
    std::span<std::byte*> blocks_span(storage->block_ptrs);
    return FixedSizedHostBuffer(
        total_size,
        block_size,
        blocks_span,
        OwningWrapper(storage, VectorStorage<std::vector<std::byte>>::delete_storage)
    );
}

FixedSizedHostBuffer FixedSizedHostBuffer::from_vectors(
    std::vector<std::vector<std::byte>> vecs
) {
    if (vecs.empty()) {
        return FixedSizedHostBuffer();
    }

    size_t const block_sz = vecs[0].size();
    size_t const total_size = block_sz * vecs.size();
    RAPIDSMPF_EXPECTS(
        std::ranges::all_of(vecs, [&](auto const& v) { return v.size() == block_sz; }),
        "all vectors must be of the same size"
    );

    auto storage = new VectorStorage<std::vector<std::vector<std::byte>>>();

    storage->block_ptrs.reserve(storage->storage.size());
    std::ranges::transform(vecs, std::back_inserter(storage->block_ptrs), [](auto& v) {
        return v.data();
    });
    storage->storage = std::move(vecs);
    std::span<std::byte*> blocks_span(storage->block_ptrs);
    return FixedSizedHostBuffer(
        total_size,
        block_sz,
        std::move(blocks_span),
        OwningWrapper(storage, VectorStorage<std::vector<std::vector<std::byte>>>::delete_storage)
    );
}

FixedSizedHostBuffer FixedSizedHostBuffer::from_multi_blocks_alloc(
    cucascade::memory::fixed_multiple_blo+cks_allocation allocation
) {
    if (!allocation || allocation->size() == 0) {
        return FixedSizedHostBuffer();
    }
    auto storage = allocation->release();
    std::span<std::byte*> blocks = shared->get_blocks();
    std::size_t total_bytes = shared->size_bytes();
    std::size_t block_sz = shared->block_size();
    auto* payload = new StoragePayload{std::shared_ptr<void>(shared)};
    return FixedSizedHostBuffer(
        total_bytes, block_sz, blocks, OwningWrapper(payload, &delete_storage_payload)
    );
}

void FixedSizedHostBuffer::reset() noexcept {
    storage_ = {};
    total_size_ = 0;
    block_size_ = 0;
    block_ptrs_ = {};
}

FixedSizedHostBuffer::FixedSizedHostBuffer(FixedSizedHostBuffer&& other) noexcept
    : storage_(std::move(other.storage_)),
      total_size_(other.total_size_),
      block_size_(other.block_size_),
      block_ptrs_(other.block_ptrs_) {
    other.reset();
}

FixedSizedHostBuffer& FixedSizedHostBuffer::operator=(FixedSizedHostBuffer&& other
) noexcept {
    storage_ = std::move(other.storage_);
    total_size_ = other.total_size_;
    block_size_ = other.block_size_;
    block_ptrs_ = other.block_ptrs_;
    other.reset();
    return *this;
}

std::span<std::byte> FixedSizedHostBuffer::block_data(std::size_t i) {
    RAPIDSMPF_EXPECTS(
        i < num_blocks(), "FixedSizedHostBuffer::block_data", std::out_of_range
    );
    return std::span<std::byte>{block_ptrs_[i], block_size_};
}

std::span<std::byte const> FixedSizedHostBuffer::block_data(std::size_t i) const {
    RAPIDSMPF_EXPECTS(
        i < num_blocks(), "FixedSizedHostBuffer::block_data", std::out_of_range
    );
    return std::span<std::byte const>{block_ptrs_[i], block_size_};
}

}  // namespace rapidsmpf
