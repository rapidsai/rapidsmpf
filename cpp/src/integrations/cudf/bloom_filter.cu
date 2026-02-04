/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include <cuda_runtime_api.h>

#include <cuco/bloom_filter_policies.cuh>
#include <cuco/bloom_filter_ref.cuh>
#include <cuco/hash_functions.cuh>
#include <cuco/utility/cuda_thread_scope.cuh>

#include <cub/device/device_transform.cuh>
#include <cuda/std/tuple>

#include <cudf/hashing.hpp>
#include <cudf/table/table_view.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/integrations/cudf/bloom_filter.hpp>
#include <rapidsmpf/nvtx.hpp>

namespace rapidsmpf {

namespace {
using KeyType = std::uint64_t;

using BloomFilterRefType = cuco::bloom_filter_ref<
    KeyType,
    cuco::extent<std::size_t>,
    cuco::thread_scope_device,
    cuco::arrow_filter_policy<KeyType, cuco::identity_hash>>;
using StorageType = BloomFilterRefType::filter_block_type;

}  // namespace

BloomFilter::BloomFilter(
    std::size_t num_blocks,
    std::uint64_t seed,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr
)
    : num_blocks_{num_blocks},
      seed_{seed},
      storage_{num_blocks * sizeof(StorageType), stream, mr} {
    // TODO: use an aligned allocator adaptor to ensure this holds.
    // Today all RMM device allocators guarantee at least 256 byte alignment, but that is
    // an implementation detail.
    RAPIDSMPF_EXPECTS(
        reinterpret_cast<std::uintptr_t>(storage_.data())
                % std::alignment_of_v<StorageType>
            == 0,
        "Allocation for bloom filter is not aligned."
    );
    RAPIDSMPF_CUDA_TRY(
        cudaMemsetAsync(storage_.data(), 0, storage_.size(), storage_.stream())
    );
}

void BloomFilter::add(
    cudf::table_view const& values_to_hash,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr
) {
    RAPIDSMPF_NVTX_FUNC_RANGE();
    auto filter_ref = BloomFilterRefType{
        static_cast<StorageType*>(storage_.data()),
        num_blocks_,
        cuco::thread_scope_device,
        {}
    };
    auto hashes = cudf::hashing::xxhash_64(values_to_hash, seed_, stream, mr);
    auto hash_view = hashes->view();
    RAPIDSMPF_EXPECTS(
        hash_view.type().id() == cudf::type_to_id<KeyType>(),
        "Hash values do not have correct type"
    );
    filter_ref.add_async(hash_view.begin<KeyType>(), hash_view.end<KeyType>(), stream);
}

void BloomFilter::merge(BloomFilter& other, rmm::cuda_stream_view stream) {
    RAPIDSMPF_NVTX_FUNC_RANGE();
    RAPIDSMPF_EXPECTS(
        num_blocks_ == other.num_blocks_, "Mismatching number of blocks in filters"
    );
    auto ref_this = BloomFilterRefType{
        static_cast<StorageType*>(storage_.data()),
        num_blocks_,
        cuco::thread_scope_device,
        {}
    };
    auto ref_other = BloomFilterRefType{
        static_cast<StorageType*>(other.storage_.data()),
        num_blocks_,
        cuco::thread_scope_device,
        {}
    };
    ref_this.merge_async(ref_other, stream);
}

rmm::device_uvector<bool> BloomFilter::contains(
    cudf::table_view const& values,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr
) {
    RAPIDSMPF_NVTX_FUNC_RANGE();
    auto filter_ref = BloomFilterRefType{
        static_cast<StorageType*>(storage_.data()),
        num_blocks_,
        cuco::thread_scope_device,
        {}
    };
    auto hashes = cudf::hashing::xxhash_64(values, seed_, stream, mr);
    auto view = hashes->view();
    rmm::device_uvector<bool> result{static_cast<std::size_t>(view.size()), stream, mr};
    filter_ref.contains_async(
        view.begin<KeyType>(), view.end<KeyType>(), result.begin(), stream
    );
    return result;
}

std::size_t BloomFilter::fitting_num_blocks(std::size_t l2size) noexcept {
    return (l2size * 2) / (3 * sizeof(StorageType));
}

rmm::cuda_stream_view BloomFilter::stream() const noexcept {
    return storage_.stream();
}

void* BloomFilter::data() noexcept {
    return storage_.data();
}

std::size_t BloomFilter::size() const noexcept {
    return storage_.size();
}

}  // namespace rapidsmpf
