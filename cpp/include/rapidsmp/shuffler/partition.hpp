/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#pragma once

#include <memory>
#include <vector>

#include <cudf/contiguous_split.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/table/table.hpp>

namespace rapidsmp::shuffler {


/**
 * @brief Partition ID, which goes from 0 to the total number of partitions in the
 * shuffle.
 *
 * The @ref PartID is always referring to a partition globally.
 */
using PartID = std::uint32_t;


/**
 * @brief Partitions rows from the input table into multiple output tables.
 *
 * @see cudf::hash_partition
 * @see cudf::split
 *
 * @throw std::out_of_range if index is `columns_to_hash` is invalid
 *
 * @param table The table to partition.
 * @param columns_to_hash Indices of input columns to hash.
 * @param num_partitions The number of partitions to use.
 * @param hash_function Hash function to use.
 * @param seed Seed value to the hash function.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned table's device memory.
 *
 * @return A vector of each partition and a table that owns the device memory.
 */
[[nodiscard]] std::pair<std::vector<cudf::table_view>, std::unique_ptr<cudf::table>>
partition_and_split(
    cudf::table_view const& table,
    std::vector<cudf::size_type> const& columns_to_hash,
    int num_partitions,
    cudf::hash_id hash_function = cudf::hash_id::HASH_MURMUR3,
    uint32_t seed = cudf::DEFAULT_HASH_SEED,
    rmm::cuda_stream_view stream = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()
);


/**
 * @brief Partitions rows from the input table into multiple packed (serialized) tables.
 *
 * @see unpack_and_concat
 * @see cudf::hash_partition
 * @see cudf::pack
 *
 * @throw std::out_of_range if index is `columns_to_hash` is invalid
 *
 * @param table The table to partition.
 * @param columns_to_hash Indices of input columns to hash.
 * @param num_partitions The number of partitions to use.
 * @param hash_function Hash function to use.
 * @param seed Seed value to the hash function.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned table's device memory.
 *
 * @return A map of partition IDs and their packed tables.
 */
[[nodiscard]] std::unordered_map<PartID, cudf::packed_columns> partition_and_pack(
    cudf::table_view const& table,
    std::vector<cudf::size_type> const& columns_to_hash,
    int num_partitions,
    cudf::hash_id hash_function = cudf::hash_id::HASH_MURMUR3,
    uint32_t seed = cudf::DEFAULT_HASH_SEED,
    rmm::cuda_stream_view stream = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()
);

/**
 * @brief Unpack (deserialize) input tables and concatenate them.
 *
 * @see partition_and_pack
 * @see cudf::unpack
 * @see cudf::concatenate
 *
 * @param partition The input tables.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned table's device memory.
 *
 * @return The unpacked and concatenated result.
 */
[[nodiscard]] std::unique_ptr<cudf::table> unpack_and_concat(
    std::vector<cudf::packed_columns>&& partition,
    rmm::cuda_stream_view stream = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()
);


}  // namespace rapidsmp::shuffler
