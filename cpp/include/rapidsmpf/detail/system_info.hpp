/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <algorithm>
#include <cstddef>
#include <ranges>

namespace rapidsmpf::detail {

/**
 * @brief Number of GPUs that share the host memory of a NUMA node.
 *
 * Counts the GPUs on `numa_node` together with the GPUs whose NUMA affinity is unknown
 * (`numa_node == -1`). If that count is zero, returns the number of all GPUs in `gpus`.
 *
 * @note In the fallback case the caller divides the memory of `numa_node` by every
 * visible GPU, which is 1 in the common one-GPU-per-process deployment. The result is
 * then not a correct per-GPU share: several unbound processes whose GPUs sit on
 * different NUMA nodes can land on the same CPU node, and each one sizes its pool
 * against that node's full memory. `PinnedPoolProperties::numa_id` defaults to the
 * same CPU node.
 *
 * @param gpus GPUs to count; each element has an `int numa_node` member.
 * @param numa_node NUMA node whose memory is being shared.
 * @return Number of GPUs to divide the node's memory by, or 0 if `gpus` is empty.
 */
template <std::ranges::sized_range Gpus>
std::size_t gpu_count_for_host_memory(Gpus const& gpus, int numa_node) {
    auto const local_count = std::ranges::count_if(gpus, [numa_node](auto const& gpu) {
        return gpu.numa_node == numa_node || gpu.numa_node == -1;
    });
    return local_count > 0 ? static_cast<std::size_t>(local_count)
                           : std::ranges::size(gpus);
}

}  // namespace rapidsmpf::detail
