/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <utility>

#include <cudf/concatenate.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/contiguous_split.hpp>  // `cudf::detail::pack` (stream ordered version)
#include <rmm/device_buffer.hpp>

#include <rapidsmpf/buffer/packed_data.hpp>
#include <rapidsmpf/buffer/resource.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/nvtx.hpp>
#include <rapidsmpf/shuffler/partition.hpp>
#include <rapidsmpf/utils.hpp>

namespace rapidsmpf::shuffler {

std::pair<std::vector<cudf::table_view>, std::unique_ptr<cudf::table>>
partition_and_split(
    cudf::table_view const& table,
    std::vector<cudf::size_type> const& columns_to_hash,
    int num_partitions,
    cudf::hash_id hash_function,
    uint32_t seed,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr
) {
    if (table.num_rows() == 0) {
        // Return views of a copy of the empty `table`.
        auto owner = std::make_unique<cudf::table>(table, stream, mr);
        return {
            std::vector<cudf::table_view>(
                static_cast<std::size_t>(num_partitions), owner->view()
            ),
            std::move(owner)
        };
    }

    auto res = cudf::hash_partition(
        table, columns_to_hash, num_partitions, hash_function, seed, stream, mr
    );
    std::unique_ptr<cudf::table> partition_table;
    partition_table.swap(res.first);

    // Notice, the offset argument for split() and hash_partition() doesn't align.
    // hash_partition() returns the start offset of each partition thus we have to
    // skip the first offset. See: <https://github.com/rapidsai/cudf/issues/4607>.
    auto partition_offsets = std::vector<int>(res.second.begin() + 1, res.second.end());

    auto tbl_partitioned =
        cudf::split(partition_table->view(), partition_offsets, stream);

    return std::make_pair(std::move(tbl_partitioned), std::move(partition_table));
}

std::unordered_map<PartID, PackedData> partition_and_pack(
    cudf::table_view const& table,
    std::vector<cudf::size_type> const& columns_to_hash,
    int num_partitions,
    cudf::hash_id hash_function,
    uint32_t seed,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr
) {
    RAPIDSMP_NVTX_FUNC_RANGE();
    auto [tables, owner] = partition_and_split(
        table, columns_to_hash, num_partitions, hash_function, seed, stream, mr
    );
    std::unordered_map<PartID, PackedData> ret;
    for (PartID i = 0; static_cast<std::size_t>(i) < tables.size(); ++i) {
        auto pack = cudf::detail::pack(tables[i], stream, mr);
        ret.emplace(i, PackedData(std::move(pack.metadata), std::move(pack.gpu_data)));
    }
    return ret;
}

std::unique_ptr<cudf::table> unpack_and_concat(
    std::vector<PackedData>&& partitions,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr
) {
    RAPIDSMP_NVTX_FUNC_RANGE();
    std::vector<cudf::table_view> unpacked;
    std::vector<cudf::packed_columns> references;
    unpacked.reserve(partitions.size());
    references.reserve(partitions.size());
    for (auto& packed_data : partitions) {
        RAPIDSMP_EXPECTS(
            (!packed_data.metadata) == (!packed_data.gpu_data),
            "the metadata and gpu_data pointers cannot be null and non-null",
            std::invalid_argument
        );
        if (packed_data.metadata) {
            unpacked.push_back(cudf::unpack(references.emplace_back(
                std::move(packed_data.metadata), std::move(packed_data.gpu_data)
            )));
        }
    }
    return cudf::concatenate(unpacked, stream, mr);
}


}  // namespace rapidsmpf::shuffler
