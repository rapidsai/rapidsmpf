/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <algorithm>
#include <cstddef>
#include <ranges>

namespace rapidsmpf::detail {

template <std::ranges::sized_range Gpus>
std::size_t gpu_count_for_host_memory(Gpus const& gpus, int numa_node) {
    auto const local_count = std::ranges::count_if(gpus, [numa_node](auto const& gpu) {
        return gpu.numa_node == numa_node || gpu.numa_node == -1;
    });
    return local_count > 0 ? static_cast<std::size_t>(local_count)
                           : std::ranges::size(gpus);
}

}  // namespace rapidsmpf::detail
