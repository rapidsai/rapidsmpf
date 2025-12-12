/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <cuco/bloom_filter_policies.cuh>
#include <cuco/bloom_filter_ref.cuh>
#include <cuco/hash_functions.cuh>
#include <cuco/utility/cuda_thread_scope.cuh>

#include <cub/device/device_transform.cuh>
#include <cuda/std/atomic>
#include <cuda/std/tuple>

#include <cudf/column/column.hpp>
#include <cudf/hashing.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>
#include <rmm/resource_ref.hpp>

#include <rapidsmpf/nvtx.hpp>

#include "bloom_filter_impl.hpp"

namespace rapidsmpf::ndsh {

namespace {
using KeyType = std::uint64_t;

using PolicyType = cuco::arrow_filter_policy<KeyType, cuco::identity_hash>;
using BloomFilterRefType = cuco::bloom_filter_ref<
    KeyType,
    cuco::extent<std::size_t>,
    cuco::thread_scope_device,
    PolicyType>;
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
      storage_{
          num_blocks * sizeof(StorageType), std::alignment_of_v<StorageType>, stream, mr
      } {
    RAPIDSMPF_CUDA_TRY(cudaMemsetAsync(storage_.data, 0, storage_.size, stream));
}

void BloomFilter::add(
    cudf::table_view const& values_to_hash,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr
) {
    NVTX3_FUNC_RANGE_IN(rapidsmpf_domain);
    auto filter_ref = BloomFilterRefType{
        static_cast<StorageType*>(storage_.data),
        num_blocks_,
        cuco::thread_scope_device,
        PolicyType{}
    };
    auto hashes = cudf::hashing::xxhash_64(values_to_hash, seed_, stream, mr);
    auto view = hashes->view();
    RAPIDSMPF_EXPECTS(
        view.type().id() == cudf::type_to_id<KeyType>(),
        "Hash values do not have correct type"
    );
    filter_ref.add_async(view.begin<KeyType>(), view.end<KeyType>(), stream);
}

void BloomFilter::merge(BloomFilter const& other, rmm::cuda_stream_view stream) {
    NVTX3_FUNC_RANGE_IN(rapidsmpf_domain);
    RAPIDSMPF_EXPECTS(
        num_blocks_ == other.num_blocks_, "Mismatching number of blocks in filters"
    );
    auto ref_this = BloomFilterRefType{
        static_cast<StorageType*>(storage_.data),
        num_blocks_,
        cuco::thread_scope_device,
        PolicyType{}
    };
    auto ref_other = BloomFilterRefType{
        static_cast<StorageType*>(other.storage_.data),
        num_blocks_,
        cuco::thread_scope_device,
        PolicyType{}
    };
    using word_type = BloomFilterRefType::word_type;
    RAPIDSMPF_CUDA_TRY(
        cub::DeviceTransform::Transform(
            cuda::std::tuple{ref_this.data(), ref_other.data()},
            ref_this.data(),
            num_blocks_ * BloomFilterRefType::words_per_block,
            [] __device__(word_type left, word_type right) { return left | right; },
            stream.value()
        )
    );
}

rmm::device_uvector<bool> BloomFilter::contains(
    cudf::table_view const& values,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr
) {
    NVTX3_FUNC_RANGE_IN(rapidsmpf_domain);
    auto filter_ref = BloomFilterRefType{
        static_cast<StorageType*>(storage_.data),
        num_blocks_,
        cuco::thread_scope_device,
        PolicyType{}
    };
    auto hashes = cudf::hashing::xxhash_64(values, seed_, stream, mr);
    auto view = hashes->view();
    rmm::device_uvector<bool> result{static_cast<std::size_t>(view.size()), stream, mr};
    filter_ref.contains_async(
        view.begin<KeyType>(), view.end<KeyType>(), result.begin(), stream
    );
    return result;
}

std::size_t BloomFilter::fitting_num_blocks(std::size_t l2size) {
    return (l2size * 2) / (3 * sizeof(StorageType));
}

rmm::cuda_stream_view BloomFilter::stream() const noexcept {
    return storage_.stream;
}

void* BloomFilter::data() const noexcept {
    return storage_.data;
}

std::size_t BloomFilter::size() const noexcept {
    return storage_.size;
}

}  // namespace rapidsmpf::ndsh
