/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

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
