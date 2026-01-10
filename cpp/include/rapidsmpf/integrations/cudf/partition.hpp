/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include <cudf/contiguous_split.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/table/table.hpp>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
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
 * @param allow_overbooking If true, allow overbooking (true by default)
 *
 * @return A vector of each partition and a table that owns the device memory.
 *
 * @throws std::out_of_range if index is `columns_to_hash` is invalid
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
    std::shared_ptr<Statistics> statistics = Statistics::disabled(),
    bool allow_overbooking = true
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
 * @param allow_overbooking If true, allow overbooking (true by default)
 * // TODO: disable this by default https://github.com/rapidsmpf/rapidsmpf/issues/449
 *
 * @return A map of partition IDs and their packed tables.
 *
 * @throws std::out_of_range if index is `columns_to_hash` is invalid
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
    std::shared_ptr<Statistics> statistics = Statistics::disabled(),
    bool allow_overbooking = true
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
 * @param allow_overbooking If true, allow overbooking (true by default)
 * // TODO: disable this by default https://github.com/rapidsmpf/rapidsmpf/issues/449
 *
 * @return A map of partition IDs and their packed tables.
 *
 * @throws std::out_of_range if the splits are invalid.
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
    std::shared_ptr<Statistics> statistics = Statistics::disabled(),
    bool allow_overbooking = true
);


/**
 * @brief Unpack (deserialize) input partitions and concatenate them into a single table.
 *
 * Empty partitions are ignored.
 *
 * The unpacking of each partition is stream-ordered on that partition's own CUDA stream.
 * The returned table is stream-ordered on the provided @p stream and synchronized with
 * the unpacking.
 *
 * @param partitions Packed input tables (partitions).
 * @param stream CUDA stream on which concatenation occurs and on which the resulting
 * table is ordered.
 * @param br Buffer resource used for memory allocations.
 * @param statistics Statistics instance to use (disabled by default).
 * @param allow_overbooking If true, allow overbooking (true by default).
 * @return The concatenated table resulting from unpacking the input partitions.
 *
 * @throws std::overflow_error If the buffer resource cannot reserve enough memory to
 * concatenate all partitions.
 * @throws std::logic_error If the partitions are not in device memory.
 *
 * @see partition_and_pack
 * @see cudf::unpack
 * @see cudf::concatenate
 */
[[nodiscard]] std::unique_ptr<cudf::table> unpack_and_concat(
    std::vector<PackedData>&& partitions,
    rmm::cuda_stream_view stream,
    BufferResource* br,
    std::shared_ptr<Statistics> statistics = Statistics::disabled(),
    bool allow_overbooking = true
);

/**
 * @brief Spill partitions from device memory to host memory.
 *
 * Moves the buffer of each `PackedData` from device memory to host memory using
 * the provided buffer resource and the buffer's CUDA stream. Partitions that are
 * already in host memory are passed through unchanged.
 *
 * For device-resident partitions, a host memory reservation is made before moving
 * the buffer. If the reservation fails due to insufficient host memory, an exception
 * is thrown. Overbooking is not allowed.
 *
 * @param partitions The partitions to spill.
 * @param br Buffer resource used to reserve host memory and perform the move.
 * @param statistics The statistics instance to use (disabled by default).
 *
 * @return A vector of `PackedData`, where each buffer resides in host memory.
 *
 * @throws std::overflow_error If host memory reservation fails.
 */
std::vector<PackedData> spill_partitions(
    std::vector<PackedData>&& partitions,
    BufferResource* br,
    std::shared_ptr<Statistics> statistics = Statistics::disabled()
);

/**
 * @brief Move spilled partitions (i.e., packed tables in host memory) back to device
 * memory.
 *
 * Each partition is inspected to determine whether its buffer resides in device memory.
 * Buffers already in device memory are left untouched. Host-resident buffers are moved
 * to device memory using the provided buffer resource and the buffer's CUDA stream.
 *
 * If insufficient device memory is available, the buffer resource's spill manager is
 * invoked to free memory. If overbooking occurs and spilling fails to reclaim enough
 * memory, behavior depends on the `allow_overbooking` flag.
 *
 * @param partitions The partitions to unspill, potentially containing host-resident data.
 * @param br Buffer resource responsible for memory reservation and spills.
 * @param allow_overbooking If false, ensures enough memory is freed to satisfy the
 * reservation; otherwise, allows overbooking even if spilling was insufficient.
 * @param statistics The statistics instance to use (disabled by default).
 *
 * @return A vector of `PackedData`, each with a buffer in device memory.
 *
 * @throws std::overflow_error If overbooking exceeds the amount spilled and
 *         `allow_overbooking` is false.
 */
std::vector<PackedData> unspill_partitions(
    std::vector<PackedData>&& partitions,
    BufferResource* br,
    bool allow_overbooking,
    std::shared_ptr<Statistics> statistics = Statistics::disabled()
);

/// @brief The amount of extra memory to reserve for packing.
constexpr size_t packing_wiggle_room_per_column = 1024;  ///< 1 KiB per column

/**
 * @brief The total amount of extra memory to reserve for packing.
 *
 * @param table The table to pack.
 * @return The total amount of extra memory to reserve for packing.
 */
inline size_t total_packing_wiggle_room(cudf::table_view const& table) {
    return packing_wiggle_room_per_column * static_cast<size_t>(table.num_columns());
}

/**
 * @brief Pack a table using a @p chunk_size device buffer using `cudf::chunked_pack`.
 *
 * All device operations will be performed on @p bounce_buf 's stream.
 * `cudf::chunked_pack` requires the buffer to be at least 1 MiB in size.
 *
 * @param table The table to pack.
 * @param bounce_buf A device bounce buffer to use for packing.
 * @param data_res Memory reservation for the data buffer. If the final packed buffer size
 * is with in a wiggle room, this @p data_res will be padded to the packed buffer size.
 *
 * @return A `PackedData` containing the packed table.
 *
 * @throws std::runtime_error If the memory allocation fails.
 * @throws std::invalid_argument If the bounce buffer is not in device memory.
 */
PackedData chunked_pack(
    cudf::table_view const& table, Buffer& bounce_buf, MemoryReservation& data_res
);

/// @brief The minimum buffer size for `cudf::chunked_pack`.
constexpr size_t cudf_chunked_pack_min_buffer_size = size_t(1) << 20;  ///< 1 MiB

/**
 * @brief Pack a table to host memory using  `cudf::pack` or `cudf::chunked_pack`.
 *
 * Based on benchmarks (rapidsai/rapidsmpf#745), the order of packing performance is as
 * follows:
 * - `cudf::pack`           -> DEVICE
 * - `cudf::chunked_pack`   -> DEVICE
 * - `cudf::pack`           -> PINNED_HOST
 * - `cudf::chunked_pack`   -> PINNED HOST
 *
 * This utility using the following strategy:
 * - data reservation must be big enough to pack the table.
 * - if the data reservation is from device accessible memory, use cudf::pack, as it
 * requires O(estimated_table_size) memory, which is already reserved up front.
 * - if the data reservation is from host memory, for each memory type in @p
 * bounce_buf_types, do the following:
 *   - try to reserve estimated_table_size for the memory type.
 *   - if the reservation is successful without overbooking, use cudf::pack, and move the
 *     packed data device buffer to the data reservation.
 *   - else if the leftover memory `>= cudf_chunked_pack_min_buffer_size`, allocate a
 *     device accessible bounce buffer, and use chunked_pack to pack to the data
 * reservation.
 *   - else loop again with the next memory type.
 *   - if all memory types are tried and no success, fail.
 *
 * @param table The table to pack.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param data_res Memory reservation for the host data buffer.
 * @param bounce_buf_types The memory types to use for the bounce buffer. Default is
 * `DEVICE_ACCESSIBLE_MEMORY_TYPES`.
 *
 * @return A `PackedData` containing the packed table.
 *
 * @throws std::invalid_argument If the memory reservation is not big enough to pack the
 * table.
 * @throws std::runtime_error If all attempts to pack the table fail.
 */
std::unique_ptr<PackedData> pack(
    cudf::table_view const& table,
    rmm::cuda_stream_view stream,
    MemoryReservation& data_res,
    std::span<MemoryType const> bounce_buf_types = DEVICE_ACCESSIBLE_MEMORY_TYPES
);

}  // namespace rapidsmpf
