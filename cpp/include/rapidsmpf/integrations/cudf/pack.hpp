/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <memory>

#include <cudf/table/table_view.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/memory_reservation.hpp>
#include <rapidsmpf/memory/memory_type.hpp>
#include <rapidsmpf/memory/packed_data.hpp>

namespace rapidsmpf {


/**
 * @brief Pack a cudf table view into a contiguous buffer using chunked packing.
 *
 * This function serializes the given table view into a `PackedData` object
 * using a bounce buffer for chunked transfer. This is useful when packing to
 * host memory to avoid allocating temporary device memory for the entire table.
 *
 * @param table The table view to pack.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param bounce_buffer Device buffer used as intermediate storage during chunked packing.
 * @param pack_temp_mr Temporary memory resource used for packing.
 * @param reservation Memory reservation to use for allocating the packed data buffer.
 * @return A unique pointer to the packed data containing the serialized table.
 *
 * @throws rapidsmpf::reservation_error If the allocation size exceeds the reservation.
 *
 * @see cudf::chunked_pack
 */
[[nodiscard]] std::unique_ptr<PackedData> chunked_pack(
    cudf::table_view const& table,
    rmm::cuda_stream_view stream,
    rmm::device_buffer& bounce_buffer,
    rmm::device_async_resource_ref pack_temp_mr,
    MemoryReservation& reservation
);

namespace detail {

/**
 * @brief Pack a cudf table view into a contiguous buffer of the specified memory type.
 *
 * - Device:
 * Uses cudf::pack(). Returns a `Buffer` with a `rmm::device_buffer`.
 *
 * - Pinned Host:
 * Uses cudf::pack() with a pinned mr as device mr. Returns a `Buffer` with a pinned
 * `HostBuffer`.
 *
 * - Host:
 * Uses cudf::chunked_pack() using a host memory reservation. Returns a `Buffer` with
 * a `HostBuffer`.
 *
 * This function serializes the given table view into a `PackedData` object
 * with the data buffer residing in the memory type specified by the template parameter.
 * The memory for the packed data is allocated using the provided reservation.
 *
 * @tparam Destination The destination memory type for the packed data buffer.
 * @param table The table view to pack.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param reservation Memory reservation to use for allocating the packed data buffer.
 *                    Must match the destination memory type.
 * @return A unique pointer to the packed data containing the serialized table.
 *
 * @throws std::invalid_argument If the reservation's memory type does not match
 * Destination.
 * @throws rapidsmpf::reservation_error If the allocation size exceeds the reservation.
 *
 * @see rapidsmpf::pack
 * @see cudf::pack
 */
template <MemoryType Destination>
[[nodiscard]] std::unique_ptr<PackedData> pack(
    cudf::table_view const& table,
    rmm::cuda_stream_view stream,
    MemoryReservation& reservation
);

}  // namespace detail

/**
 * @brief Pack a cudf table view into a contiguous buffer.
 *
 * This function serializes the given table view into a `PackedData` object
 * with the data buffer residing in the memory type of the provided reservation.
 * The memory for the packed data is allocated using the provided reservation.
 *
 * @param table The table view to pack.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param reservation Memory reservation to use for allocating the packed data buffer.
 * @return A unique pointer to the packed data containing the serialized table.
 *
 * @throws rapidsmpf::reservation_error If the allocation size exceeds the reservation.
 *
 * @see cudf::pack
 */
[[nodiscard]] std::unique_ptr<PackedData> pack(
    cudf::table_view const& table,
    rmm::cuda_stream_view stream,
    MemoryReservation& reservation
);

}  // namespace rapidsmpf
