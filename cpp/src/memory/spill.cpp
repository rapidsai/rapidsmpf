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
    // Spill each partition to host memory.
    auto reservation = br->reserve_or_fail(device_size, SPILL_TARGET_MEMORY_TYPES);
    std::vector<PackedData> ret;
    ret.reserve(partitions.size());
    for (auto& [metadata, data] : partitions) {
        ret.emplace_back(std::move(metadata), br->move(std::move(data), reservation));
    }
    return ret;
}

std::vector<PackedData> unspill_partitions(
    std::vector<PackedData>&& partitions,
    BufferResource* br,
    AllowOverbooking allow_overbooking
) {
    auto statistics = br->statistics();
    // Sum the total size of all packed data not in device memory already.
    std::size_t non_device_size{0};
    for (auto& [_, data] : partitions) {
        if (data->mem_type() != MemoryType::DEVICE) {
            non_device_size += data->size;
        }
    }

    // Unspill each partition.
    auto reservation =
        br->reserve_device_memory_and_spill(non_device_size, allow_overbooking);
    std::vector<PackedData> ret;
    ret.reserve(partitions.size());
    for (auto& [metadata, data] : partitions) {
        ret.emplace_back(std::move(metadata), br->move(std::move(data), reservation));
    }

    return ret;
}

}  // namespace rapidsmpf
