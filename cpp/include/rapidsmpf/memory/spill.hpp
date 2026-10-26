/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <memory>
#include <span>
#include <vector>

#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/packed_data.hpp>

namespace rapidsmpf {

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
 *
 * @return A vector of `PackedData`, where each buffer resides in host memory.
 *
 * @throws rapidsmpf::reservation_error If host memory reservation fails.
 */
std::vector<PackedData> spill_partitions(
    std::vector<PackedData>&& partitions, BufferResource* br
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
 *
 * @return A vector of `PackedData`, each with a buffer in device memory.
 *
 * @throws rapidsmpf::reservation_error If overbooking exceeds the amount spilled and
 *         `allow_overbooking` is false.
 */
std::vector<PackedData> unspill_partitions(
    std::vector<PackedData>&& partitions,
    BufferResource* br,
    AllowOverbooking allow_overbooking
);

/**
 * @brief Move a buffer into the last memory type of @p accepted, unless it is already
 * in one of @p accepted.
 *
 * The last memory type is overbooked if it doesn't fit and
 * `allow_overbooking_by_default()` of @p br allows it. The move may block the calling
 * thread.
 *
 * @param buffer The buffer. Must not be `nullptr`.
 * @param accepted The accepted memory types in decreasing order of preference. Must
 * not be empty.
 * @param br Buffer resource used to reserve memory and perform the move.
 * @return @p buffer, unchanged if it is already in one of @p accepted, otherwise moved
 * into the last memory type of @p accepted.
 *
 * @throws std::invalid_argument If @p buffer is `nullptr` or @p accepted is empty.
 * @throws rapidsmpf::reservation_error If @p buffer must be moved, the last memory type
 * doesn't fit, and overbooking is not allowed.
 */
[[nodiscard]] std::unique_ptr<Buffer> ensure_memory_type(
    std::unique_ptr<Buffer> buffer,
    std::span<MemoryType const> accepted,
    BufferResource* br
);

}  // namespace rapidsmpf
