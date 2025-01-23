/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cudf/concatenate.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/contiguous_split.hpp>  // `cudf::detail::pack` (stream ordered version)

#include <rapidsmp/error.hpp>
#include <rapidsmp/nvtx.hpp>
#include <rapidsmp/shuffler/partition.hpp>
#include <rapidsmp/utils.hpp>

namespace rapidsmp::shuffler {

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
        // Return an copy of the empty `table`.
        return std::
            make_pair<std::vector<cudf::table_view>, std::unique_ptr<cudf::table>>(
                {cudf::table(table, stream, mr)}, nullptr
            );
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

std::unordered_map<PartID, cudf::packed_columns> partition_and_pack(
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
    std::unordered_map<PartID, cudf::packed_columns> ret;
    for (PartID i = 0; static_cast<std::size_t>(i) < tables.size(); ++i) {
        ret[i] = cudf::detail::pack(tables[i], stream, mr);
    }
    return ret;
}

std::unique_ptr<cudf::table> unpack_and_concat(
    std::vector<cudf::packed_columns>&& partitions,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr
) {
    RAPIDSMP_NVTX_FUNC_RANGE();
    std::vector<cudf::table_view> unpacked;
    unpacked.reserve(partitions.size());
    for (auto const& packed_columns : partitions) {
        RAPIDSMP_EXPECTS(
            (!packed_columns.metadata) == (!packed_columns.gpu_data),
            "the metadata and gpu_data pointers cannot be null and non-null",
            std::invalid_argument
        );
        if (packed_columns.metadata) {
            unpacked.push_back(cudf::unpack(packed_columns));
        }
    }
    return cudf::concatenate(unpacked, stream, mr);
}


}  // namespace rapidsmp::shuffler
