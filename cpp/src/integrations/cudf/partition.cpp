/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <utility>

#include <cudf/concatenate.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/copying.hpp>
#include <cudf/types.hpp>
#include <rmm/device_buffer.hpp>

#include <rapidsmpf/buffer/packed_data.hpp>
#include <rapidsmpf/buffer/resource.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/integrations/cudf/partition.hpp>
#include <rapidsmpf/nvtx.hpp>
#include <rapidsmpf/utils.hpp>

namespace rapidsmpf {

std::pair<std::vector<cudf::table_view>, std::unique_ptr<cudf::table>>
partition_and_split(
    cudf::table_view const& table,
    std::vector<cudf::size_type> const& columns_to_hash,
    int num_partitions,
    cudf::hash_id hash_function,
    uint32_t seed,
    rmm::cuda_stream_view stream,
    BufferResource* br,
    std::shared_ptr<Statistics> statistics
) {
    RAPIDSMPF_MEMORY_PROFILE(statistics);
    if (table.num_rows() == 0) {
        // Return views of a copy of the empty `table`.
        auto owner = std::make_unique<cudf::table>(table, stream, br->device_mr());
        return {
            std::vector<cudf::table_view>(
                static_cast<std::size_t>(num_partitions), owner->view()
            ),
            std::move(owner)
        };
    }

    auto res = cudf::hash_partition(
        table,
        columns_to_hash,
        num_partitions,
        hash_function,
        seed,
        stream,
        br->device_mr()
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

std::unordered_map<shuffler::PartID, PackedData> partition_and_pack(
    cudf::table_view const& table,
    std::vector<cudf::size_type> const& columns_to_hash,
    int num_partitions,
    cudf::hash_id hash_function,
    uint32_t seed,
    rmm::cuda_stream_view stream,
    BufferResource* br,
    std::shared_ptr<Statistics> statistics
) {
    RAPIDSMPF_NVTX_FUNC_RANGE();
    RAPIDSMPF_MEMORY_PROFILE(statistics);
    RAPIDSMPF_EXPECTS(num_partitions > 0, "Need to split to at least one partition");
    if (table.num_rows() == 0) {
        auto splits = std::vector<cudf::size_type>(
            static_cast<std::uint64_t>(num_partitions - 1), 0
        );
        return split_and_pack(table, splits, stream, br);
    }
    auto [reordered, split_points] = cudf::hash_partition(
        table,
        columns_to_hash,
        num_partitions,
        hash_function,
        seed,
        stream,
        br->device_mr()
    );
    std::vector<cudf::size_type> splits(split_points.begin() + 1, split_points.end());
    return split_and_pack(reordered->view(), splits, stream, br);
}

std::unordered_map<shuffler::PartID, PackedData> split_and_pack(
    cudf::table_view const& table,
    std::vector<cudf::size_type> const& splits,
    rmm::cuda_stream_view stream,
    BufferResource* br,
    std::shared_ptr<Statistics> statistics
) {
    RAPIDSMPF_NVTX_FUNC_RANGE();
    RAPIDSMPF_MEMORY_PROFILE(statistics);
    std::unordered_map<shuffler::PartID, PackedData> ret;
    auto packed = cudf::contiguous_split(table, splits, stream, br->device_mr());
    for (shuffler::PartID i = 0; static_cast<std::size_t>(i) < packed.size(); i++) {
        auto pack = std::move(packed[i].data);
        ret.emplace(
            i,
            PackedData(
                std::move(pack.metadata), br->move(std::move(pack.gpu_data), stream)
            )
        );
    }
    return ret;
}

std::unique_ptr<cudf::table> unpack_and_concat(
    std::vector<PackedData>&& partitions,
    rmm::cuda_stream_view stream,
    BufferResource* br,
    std::shared_ptr<Statistics> statistics
) {
    RAPIDSMPF_NVTX_FUNC_RANGE();
    RAPIDSMPF_MEMORY_PROFILE(statistics);
    std::vector<cudf::table_view> unpacked;
    std::vector<cudf::packed_columns> references;
    unpacked.reserve(partitions.size());
    references.reserve(partitions.size());
    for (auto& packed_data : partitions) {
        if (!packed_data.empty()) {
            unpacked.push_back(
                cudf::unpack(references.emplace_back(
                    std::move(packed_data.metadata),
                    br->move_to_device_buffer(std::move(packed_data.gpu_data))
                ))
            );
        }
    }
    return cudf::concatenate(unpacked, stream, br->device_mr());
}


}  // namespace rapidsmpf
