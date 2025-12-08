/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <cstdint>

#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <cuco/bloom_filter.cuh>
#include <cuco/bloom_filter_policies.cuh>
#include <cuco/hash_functions.cuh>
#include <cuco/utility/cuda_thread_scope.cuh>

#include <cuda/std/atomic>

#include <cudf/column/column.hpp>
#include <cudf/hashing.hpp>
#include <cudf/table/table_view.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>
#include <rmm/resource_ref.hpp>

#include "bloom_filter_impl.hpp"

namespace rapidsmpf::ndsh {

using policy_type =
    cuco::default_filter_policy<cuco::identity_hash<std::uint64_t>, std::uint32_t, 8>;
using bloom_filter = cuco::bloom_filter<
    std::uint64_t,
    cuco::extent<std::size_t>,
    cuda::thread_scope_device,
    policy_type,
    rmm::mr::polymorphic_allocator<char>>;

using bloom_filter_ref = bloom_filter::ref_type<cuco::thread_scope_device>;

aligned_buffer create_filter_storage(
    std::size_t num_blocks,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr
) {
    using type = bloom_filter_ref::filter_block_type;
    return aligned_buffer{
        num_blocks * sizeof(type), std::alignment_of_v<type>, stream, mr
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
    auto policy = policy_type{};
    auto filter_ref = bloom_filter_ref(
        static_cast<bloom_filter_ref::filter_block_type*>(storage.data),
        num_blocks,
        cuco::thread_scope_device,
        policy
    );
    auto hashes = cudf::hashing::xxhash_64(values_to_hash, seed, stream, mr);
    auto view = hashes->view();
    filter_ref.add_async(view.begin<std::uint64_t>(), view.end<std::uint64_t>(), stream);
}

rmm::device_uvector<bool> apply_filter(
    aligned_buffer& storage,
    std::size_t num_blocks,
    cudf::table_view const& values_to_hash,
    std::uint64_t seed,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr
) {
    auto policy = policy_type{};
    auto filter_ref = bloom_filter_ref(
        static_cast<bloom_filter_ref::filter_block_type*>(storage.data),
        num_blocks,
        cuco::thread_scope_device,
        policy
    );
    auto hashes = cudf::hashing::xxhash_64(values_to_hash, seed, stream, mr);
    auto view = hashes->view();
    rmm::device_uvector<bool> result(static_cast<std::size_t>(view.size()), stream, mr);
    filter_ref.contains_async(
        view.begin<std::uint64_t>(), view.end<std::uint64_t>(), result.begin(), stream
    );
    return result;
}
}  // namespace rapidsmpf::ndsh
