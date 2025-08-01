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

#include <rapidsmpf/buffer/buffer.hpp>
#include <rapidsmpf/buffer/packed_data.hpp>
#include <rapidsmpf/buffer/resource.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/shuffler/shuffler.hpp>
#include <rapidsmpf/statistics.hpp>

namespace rapidsmpf {


/**
 * @brief Partitions rows from the input table into multiple output tables.
 *
 * @param table The table to partition.
 * @param columns_to_hash Indices of input columns to hash.
 * @param num_partitions The number of partitions.
 * @param hash_function Hash function to use.
 * @param seed Seed value to the hash function.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param br Buffer resource for memory allocations.
 * @param statistics The statistics instance to use (disabled by default).
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
    BufferResource* br,
    std::shared_ptr<Statistics> statistics = Statistics::disabled()
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
 * @param br Buffer resource for memory allocations.
 * @param statistics The statistics instance to use (disabled by default).
 *
 * @return A map of partition IDs and their packed tables.
 *
 * @throw std::out_of_range if index is `columns_to_hash` is invalid
 *
 * @see unpack_and_concat
 * @see cudf::hash_partition
 * @see cudf::pack
 */
[[nodiscard]] std::unordered_map<shuffler::PartID, PackedData> partition_and_pack(
    cudf::table_view const& table,
    std::vector<cudf::size_type> const& columns_to_hash,
    int num_partitions,
    cudf::hash_id hash_function,
    uint32_t seed,
    rmm::cuda_stream_view stream,
    BufferResource* br,
    std::shared_ptr<Statistics> statistics = Statistics::disabled()
);


/**
 * @brief Splits rows from the input table into multiple packed (serialized) tables.
 *
 * @param table The table to split and pack into partitions.
 * @param splits The split points, equivalent to cudf::split(), i.e. one less than
 * the number of result partitions.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param br Buffer resource for memory allocations.
 * @param statistics The statistics instance to use (disabled by default).
 *
 * @return A map of partition IDs and their packed tables.
 *
 * @throw std::out_of_range if the splits are invalid.
 *
 * @see unpack_and_concat
 * @see cudf::split
 * @see partition_and_pack
 */
[[nodiscard]] std::unordered_map<shuffler::PartID, PackedData> split_and_pack(
    cudf::table_view const& table,
    std::vector<cudf::size_type> const& splits,
    rmm::cuda_stream_view stream,
    BufferResource* br,
    std::shared_ptr<Statistics> statistics = Statistics::disabled()
);


/**
 * @brief Unpack (deserialize) input tables and concatenate them.
 *
 * Ignores empty partitions.
 *
 * @param partitions The packed input tables.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param br Buffer resource for memory allocations.
 * @param statistics The statistics instance to use (disabled by default).
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
    BufferResource* br,
    std::shared_ptr<Statistics> statistics = Statistics::disabled()
);

/**
 * @brief Move spilled partitions (i.e., packed tables in host memory) back to device
 * memory.
 *
 * Each partition is inspected to determine whether its buffer resides in device memory.
 * Buffers already in device memory are left untouched. Host-resident buffers are moved
 * to device memory using the provided buffer resource and CUDA stream.
 *
 * If insufficient device memory is available, the buffer resource's spill manager is
 * invoked to free memory. If overbooking occurs and spilling fails to reclaim enough
 * memory, behavior depends on the `allow_overbooking` flag.
 *
 * @param partitions The partitions to unspill, potentially containing host-resident data.
 * @param stream CUDA stream used for memory operations and kernel launches.
 * @param br Buffer resource responsible for memory reservation and spills.
 * @param allow_overbooking If false, ensures enough memory is freed to satisfy the
 * reservation; otherwise, allows overbooking even if spilling was insufficient.
 *
 * @return A vector of `PackedData`, each with a buffer in device memory.
 *
 * @throws std::overflow_error If overbooking exceeds the amount spilled and
 *         `allow_overbooking` is false.
 */
std::vector<PackedData> unspill_partitions(
    std::vector<PackedData>&& partitions,
    rmm::cuda_stream_view stream,
    BufferResource* br,
    bool allow_overbooking
) {
    // Sum the total size of all packed data not in device memory already.
    std::size_t non_device_size{0};
    for (auto& [_, data] : partitions) {
        if (data->mem_type() != MemoryType::DEVICE) {
            non_device_size += data->size;
        }
    }
    // This total sum is what we need to reserve before moving data to device.
    auto [reservation, overbooking] =
        br->reserve(MemoryType::DEVICE, non_device_size, true);

    // Check overbooking, do we need to spill to host memory?
    if (overbooking > 0) {
        std::size_t spilled = br->spill_manager().spill(overbooking);
        RAPIDSMPF_EXPECTS(
            allow_overbooking || spilled >= overbooking,
            "could not spill enough to make room to unspill all partitions",
            std::overflow_error
        );
    }
    // Unspill each partition.
    std::vector<PackedData> ret;
    ret.reserve(partitions.size());
    for (auto& [metadata, data] : partitions) {
        ret.emplace_back(
            std::move(metadata),
            br->move(MemoryType::DEVICE, std::move(data), stream, reservation)
        );
    }
    return ret;
}

}  // namespace rapidsmpf
