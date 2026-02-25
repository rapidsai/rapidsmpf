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
 * @brief Pack a cudf table view into a contiguous device buffer.
 *
 * Uses cudf::pack(). Returns a `PackedData` with a `Buffer` backed by
 * `rmm::device_buffer`. The memory is allocated using the provided reservation.
 *
 * @param table The table view to pack.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param reservation Device memory reservation. Must have memory type DEVICE.
 * @return A unique pointer to the packed data containing the serialized table.
 *
 * @throws std::invalid_argument If the reservation's memory type is not DEVICE.
 * @throws rapidsmpf::reservation_error If the allocation size exceeds the reservation.
 *
 * @see cudf::pack
 */
[[nodiscard]] std::unique_ptr<PackedData> pack_device(
    cudf::table_view const& table,
    rmm::cuda_stream_view stream,
    MemoryReservation& reservation
);

/**
 * @brief Pack a cudf table view into a contiguous pinned host buffer.
 *
 * Uses cudf::pack() with a pinned memory resource. Returns a `PackedData` with
 * a `Buffer` backed by a pinned `HostBuffer`. The memory is allocated using
 * the provided reservation.
 *
 * @param table The table view to pack.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param reservation Pinned host memory reservation. Must have memory type PINNED_HOST.
 * @return A unique pointer to the packed data containing the serialized table.
 *
 * @throws std::invalid_argument If the reservation's memory type is not PINNED_HOST.
 * @throws rapidsmpf::reservation_error If the allocation size exceeds the reservation.
 *
 * @see cudf::pack
 */
[[nodiscard]] std::unique_ptr<PackedData> pack_pinned_host(
    cudf::table_view const& table,
    rmm::cuda_stream_view stream,
    MemoryReservation& reservation
);

/**
 * @brief Pack a cudf table view into a contiguous host buffer.
 *
 * Uses cudf::chunked_pack() with a device bounce buffer when available,
 * otherwise a pinned bounce buffer. Returns a `PackedData` with a `Buffer`
 * backed by a `HostBuffer`. The memory is allocated using the provided reservation.
 *
 * Algorithm:
 * 1. Special case: empty tables return immediately with empty packed data.
 * 2. Fast path for small tables (< 1MB): pack directly on device and copy to host.
 * 3. Estimate the table size (est_size), with a minimum of 1MB.
 * 4. Try to reserve device memory for est_size with overbooking allowed.
 * 5. If available device memory (reservation - overbooking) >= 1MB,
 *    use chunked packing with the device bounce buffer.
 * 6. Otherwise, if pinned memory is available, retry with pinned memory (steps 4-5).
 * 7. If all attempts fail, throw an error.
 *
 * @param table The table view to pack.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param reservation Host memory reservation. Must have memory type HOST.
 * @return A unique pointer to the packed data containing the serialized table.
 *
 * @throws std::invalid_argument If the reservation's memory type is not HOST.
 * @throws rapidsmpf::reservation_error If the allocation size exceeds the reservation.
 *
 * @see cudf::chunked_pack
 */
[[nodiscard]] std::unique_ptr<PackedData> pack_host(
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
