/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rapidsmpf/detail/system_info.hpp>
#include <rapidsmpf/system_info.hpp>

#if RAPIDSMPF_HAVE_NUMA
#include <numa.h>
#endif

using namespace rapidsmpf;

TEST(GetCurrentNumaNodeIdTest, ReturnsValidNumaNodeId) {
    int numa_node_id = get_current_numa_node();
    EXPECT_GE(numa_node_id, 0);
#if RAPIDSMPF_HAVE_NUMA
    EXPECT_LE(numa_node_id, numa_max_node());
#endif
}

namespace {
struct TestGpu {
    int numa_node;
};
}  // namespace

TEST(HostMemoryPerGpuTest, CountsLocalAndUnknownAffinity) {
    std::array gpus{TestGpu{0}, TestGpu{0}, TestGpu{1}, TestGpu{-1}};
    EXPECT_EQ(detail::gpu_count_for_host_memory(gpus, 0), 3u);
    EXPECT_EQ(detail::gpu_count_for_host_memory(gpus, 1), 2u);
    EXPECT_EQ(detail::gpu_count_for_host_memory(gpus, 2), 1u);
}

TEST(HostMemoryPerGpuTest, FallsBackToVisibleGpusWhenNoneAreLocal) {
    std::array gpus{TestGpu{2}, TestGpu{3}, TestGpu{3}};
    EXPECT_EQ(detail::gpu_count_for_host_memory(gpus, 0), 3u);
    EXPECT_EQ(detail::gpu_count_for_host_memory(gpus, 2), 1u);
    EXPECT_EQ(detail::gpu_count_for_host_memory(gpus, 3), 2u);
}

TEST(HostMemoryPerGpuTest, ReportsZeroWhenNoGpusAreVisible) {
    std::array<TestGpu, 0> gpus{};
    EXPECT_EQ(detail::gpu_count_for_host_memory(gpus, 0), 0u);
}
