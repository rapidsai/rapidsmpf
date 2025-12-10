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

#include <cuco/bloom_filter.cuh>
#include <cuco/bloom_filter_policies.cuh>
#include <cuco/hash_functions.cuh>
#include <cuco/utility/cuda_thread_scope.cuh>

#include <cub/device/device_transform.cuh>
#include <cuda/std/atomic>

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

using KeyType = std::uint64_t;

using PolicyType = cuco::arrow_filter_policy<KeyType, cuco::identity_hash>;
using BloomFilter = cuco::bloom_filter<
    std::uint64_t,
    cuco::extent<std::size_t>,
    cuda::thread_scope_device,
    PolicyType,
    rmm::mr::polymorphic_allocator<char>>;

using BloomFilterRef = BloomFilter::ref_type<cuco::thread_scope_device>;
using StorageType = BloomFilterRef::filter_block_type;

aligned_buffer create_filter_storage(
    std::size_t num_blocks,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr
) {
    return aligned_buffer{
        num_blocks * sizeof(StorageType), std::alignment_of_v<StorageType>, stream, mr
    };
}

void update_filter(
    aligned_buffer& storage,
    std::size_t num_blocks,
    cudf::table_view const& values_to_hash,
    std::uint64_t seed,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr
) {
    RAPIDSMPF_NVTX_FUNC_RANGE();
    auto filter_ref = BloomFilterRef{
        static_cast<StorageType*>(storage.data),
        num_blocks,
        cuco::thread_scope_device,
        PolicyType{}
    };
    auto hashes = cudf::hashing::xxhash_64(values_to_hash, seed, stream, mr);
    auto view = hashes->view();
    RAPIDSMPF_EXPECTS(
        view.type().id() == cudf::type_to_id<KeyType>(),
        "Hash values do not have correct type"
    );
    filter_ref.add_async(view.begin<KeyType>(), view.end<KeyType>(), stream);
}

rmm::device_uvector<bool> apply_filter(
    aligned_buffer& storage,
    std::size_t num_blocks,
    cudf::table_view const& values_to_hash,
    std::uint64_t seed,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr
) {
    RAPIDSMPF_NVTX_FUNC_RANGE();
    auto policy = PolicyType{};
    auto filter_ref = BloomFilterRef{
        static_cast<StorageType*>(storage.data),
        num_blocks,
        cuco::thread_scope_device,
        policy
    };
    auto hashes = cudf::hashing::xxhash_64(values_to_hash, seed, stream, mr);
    auto view = hashes->view();
    rmm::device_uvector<bool> result(static_cast<std::size_t>(view.size()), stream, mr);
    filter_ref.contains_async(
        view.begin<KeyType>(), view.end<KeyType>(), result.begin(), stream
    );
    return result;
}

void merge_filters(
    aligned_buffer& storage,
    const aligned_buffer& other,
    std::size_t num_blocks,
    rmm::cuda_stream_view stream
) {
    auto ref_out = BloomFilterRef{
        static_cast<StorageType*>(storage.data),
        num_blocks,
        cuco::thread_scope_device,
        PolicyType{}
    };
    auto ref_in = BloomFilterRef{
        static_cast<StorageType*>(other.data),
        num_blocks,
        cuco::thread_scope_device,
        PolicyType{}
    };
    using word_type = BloomFilterRef::word_type;
    RAPIDSMPF_CUDA_TRY(
        cub::DeviceTransform::Transform(
            cuda::std::tuple{ref_out.data(), ref_in.data()},
            ref_out.data(),
            num_blocks * BloomFilterRef::words_per_block,
            [] __device__(word_type left, word_type right) { return left | right; },
            stream.value()
        )
    );
}

std::size_t num_filter_blocks(int l2cachesize) {
    return (static_cast<std::size_t>(l2cachesize) * 2) / (3 * sizeof(StorageType));
}
}  // namespace rapidsmpf::ndsh
