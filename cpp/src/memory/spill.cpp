/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <utility>

#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/memory_type.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/memory/spill.hpp>
#include <rapidsmpf/utils/misc.hpp>

namespace rapidsmpf {

std::vector<PackedData> spill_partitions(
    std::vector<PackedData>&& partitions, BufferResource* br
) {
    // Sum the total size of all packed data in device memory.
    std::size_t device_size{0};
    for (auto& [_, data] : partitions) {
        if (data->mem_type() == MemoryType::DEVICE) {
            device_size += data->size;
        }
    }
    // Spill device partitions to the spill target memory types.
    auto reservation = br->reserve_or_fail(device_size, SPILL_TARGET_MEMORY_TYPES);
    for (auto& [_, data] : partitions) {
        if (data->mem_type() == MemoryType::DEVICE) {
            data = br->move(std::move(data), reservation);
        }
    }
    return std::move(partitions);
}

std::vector<PackedData> unspill_partitions(
    std::vector<PackedData>&& partitions,
    BufferResource* br,
    AllowOverbooking allow_overbooking
) {
    // Sum the total size of all packed data not in device memory already.
    std::size_t non_device_size{0};
    for (auto& [_, data] : partitions) {
        if (data->mem_type() != MemoryType::DEVICE) {
            non_device_size += data->size;
        }
    }

    // Unspill non-device partitions.
    auto reservation =
        br->reserve_device_memory_and_spill(non_device_size, allow_overbooking);
    for (auto& [_, data] : partitions) {
        if (data->mem_type() != MemoryType::DEVICE) {
            data = br->move(std::move(data), reservation);
        }
    }

    return std::move(partitions);
}

std::unique_ptr<Buffer> ensure_memory_type(
    std::unique_ptr<Buffer> buffer,
    std::span<MemoryType const> accepted,
    BufferResource* br
) {
    RAPIDSMPF_EXPECTS(
        buffer != nullptr, "the buffer pointer cannot be NULL", std::invalid_argument
    );
    RAPIDSMPF_EXPECTS(
        !accepted.empty(),
        "the accepted memory types cannot be empty",
        std::invalid_argument
    );
    if (contains(accepted, buffer->mem_type())) {
        return buffer;
    }
    auto [res, _] =
        br->reserve(accepted.back(), buffer->size, br->allow_overbooking_by_default());
    RAPIDSMPF_EXPECTS(
        res.size() == buffer->size,
        "cannot reserve memory to move the buffer and overbooking is not allowed",
        rapidsmpf::reservation_error
    );
    return br->move(std::move(buffer), res);
}

}  // namespace rapidsmpf
