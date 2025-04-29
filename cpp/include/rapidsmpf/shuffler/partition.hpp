/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include <cudf/contiguous_split.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/table/table.hpp>

#include <rapidsmpf/buffer/packed_data.hpp>

namespace rapidsmpf::shuffler {


/**
 * @brief Partition ID, which goes from 0 to the total number of partitions in the
 * shuffle.
 *
 * The `PartID` is always referring to a partition globally.
 */
using PartID = std::uint32_t;


/**
 * @brief Partitions rows from the input table into multiple output tables.
 *
 * @param table The table to partition.
 * @param columns_to_hash Indices of input columns to hash.
 * @param num_partitions The number of partitions.
 * @param hash_function Hash function to use.
 * @param seed Seed value to the hash function.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned table's device memory.
 *
 * @return A vector of each partition and a table that owns the device memory.
 *
 * @throw std::out_of_range if index is `columns_to_hash` is invalid
 *
 * @see cudf::hash_partition
 * @see cudf::split
 */
[[nodiscard]] std::pair<std::vector<cudf::table_view>, std::unique_ptr<cudf::table>>
partition_and_split(
    cudf::table_view const& table,
    std::vector<cudf::size_type> const& columns_to_hash,
    int num_partitions,
    cudf::hash_id hash_function,
    uint32_t seed,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr
);


/**
 * @brief Partitions rows from the input table into multiple packed (serialized) tables.
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
 *
 * @throw std::out_of_range if index is `columns_to_hash` is invalid
 *
 * @see unpack_and_concat
 * @see cudf::hash_partition
 * @see cudf::pack
 */
[[nodiscard]] std::unordered_map<PartID, PackedData> partition_and_pack(
    cudf::table_view const& table,
    std::vector<cudf::size_type> const& columns_to_hash,
    int num_partitions,
    cudf::hash_id hash_function,
    uint32_t seed,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr
);

/**
 * @brief Unpack (deserialize) input tables and concatenate them.
 *
 * Ignores partitions with metadata and gpu_data null pointers.
 *
 * @param partitions The packed input tables.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned table's device memory.
 *
 * @return The unpacked and concatenated result.
 *
 * @see partition_and_pack
 * @see cudf::unpack
 * @see cudf::concatenate
 */
[[nodiscard]] std::unique_ptr<cudf::table> unpack_and_concat(
    std::vector<PackedData>&& partitions,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr
);


}  // namespace rapidsmpf::shuffler
