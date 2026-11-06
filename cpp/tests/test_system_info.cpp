/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <optional>
#include <thread>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rapidsmpf/system_info.hpp>

#if RAPIDSMPF_HAVE_NUMA
#include <numa.h>
#include <numaif.h>
#endif

using namespace rapidsmpf;

TEST(GetCurrentNumaNodeIdTest, ReturnsValidNumaNodeId) {
    int numa_node_id = get_current_numa_node();
    EXPECT_GE(numa_node_id, 0);
#if RAPIDSMPF_HAVE_NUMA
    EXPECT_LE(numa_node_id, numa_max_node());
#endif
}

#if RAPIDSMPF_HAVE_NUMA
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::UnorderedElementsAreArray;

/// Host (CPU-attached) NUMA nodes the process may allocate from, and nodes that have
/// memory but no CPUs (e.g. GPU HBM on GB/GH systems).
struct Topology {
    std::vector<int> host;
    std::vector<int> cpuless;
};

Topology discover_topology() {
    Topology topo;
    if (numa_available() == -1) {
        return topo;
    }
    struct bitmask* allowed = numa_get_mems_allowed();
    struct bitmask* cpus = numa_allocate_cpumask();
    for (int node = 0; node <= numa_max_node(); ++node) {
        if (numa_bitmask_isbitset(allowed, node) == 0) {
            continue;
        }
        if (numa_node_to_cpus(node, cpus) == 0 && numa_bitmask_weight(cpus) > 0) {
            topo.host.push_back(node);
        } else {
            topo.cpuless.push_back(node);
        }
    }
    numa_free_cpumask(cpus);
    numa_bitmask_free(allowed);
    return topo;
}

/**
 * @brief Calls `get_current_numa_nodes()` from a fresh thread with the given policy.
 *
 * Memory policies and CPU affinity are per-thread, so each case runs in its own thread
 * and does not leak into the rest of the test process.
 *
 * @param mode Memory policy (`MPOL_*`).
 * @param policy_nodes Nodes passed to `set_mempolicy()`.
 * @param pin_to_node If >= 0, also pin the thread's CPUs to this node.
 * @return The nodes, or `std::nullopt` if the policy could not be applied (e.g.
 * `EPERM` in a container).
 */
std::optional<std::vector<int>> nodes_with_policy(
    int mode, std::vector<int> const& policy_nodes, int pin_to_node = -1
) {
    std::optional<std::vector<int>> ret;
    std::thread([&] {
        if (pin_to_node >= 0 && numa_run_on_node(pin_to_node) != 0) {
            return;
        }
        if (mode != MPOL_DEFAULT) {
            struct bitmask* mask = numa_allocate_nodemask();
            for (int node : policy_nodes) {
                numa_bitmask_setbit(mask, node);
            }
            auto const rc = set_mempolicy(mode, mask->maskp, mask->size + 1);
            numa_free_nodemask(mask);
            if (rc != 0) {
                return;
            }
        }
        ret = get_current_numa_nodes();
    }).join();
    return ret;
}

}  // namespace

// ---------------------------------------------------------------------------
// 1 NUMA node: the default system, including CI.
// ---------------------------------------------------------------------------
class SingleNumaNodeTest : public ::testing::Test {
  protected:
    void SetUp() override {
        topo = discover_topology();
        if (topo.host.size() != 1 || !topo.cpuless.empty()) {
            GTEST_SKIP() << "requires a system with exactly one NUMA node";
        }
    }

    Topology topo;
};

TEST_F(SingleNumaNodeTest, AllPoliciesReturnTheOnlyNode) {
    auto const expected = ElementsAre(topo.host[0]);
    for (int mode : {MPOL_DEFAULT, MPOL_BIND, MPOL_PREFERRED, MPOL_INTERLEAVE}) {
        auto const nodes = nodes_with_policy(mode, topo.host);
        ASSERT_TRUE(nodes.has_value()) << "could not apply memory policy / CPU pinning";
        EXPECT_THAT(*nodes, expected) << "mode " << mode;
    }
}

// ---------------------------------------------------------------------------
// 2 NUMA nodes (e.g. a 2-socket DGX): every node has CPUs and memory.
// ---------------------------------------------------------------------------
class TwoNumaNodesTest : public ::testing::Test {
  protected:
    void SetUp() override {
        topo = discover_topology();
        if (topo.host.size() != 2 || !topo.cpuless.empty()) {
            GTEST_SKIP() << "requires a system with exactly two NUMA nodes";
        }
        n0 = topo.host[0];
        n1 = topo.host[1];
    }

    Topology topo;
    int n0{};
    int n1{};
};

TEST_F(TwoNumaNodesTest, DefaultPolicyReturnsAllNodes) {
    auto const nodes = nodes_with_policy(MPOL_DEFAULT, {});
    ASSERT_TRUE(nodes.has_value()) << "could not apply memory policy / CPU pinning";
    EXPECT_THAT(*nodes, ElementsAre(n0, n1));
}

TEST_F(TwoNumaNodesTest, DefaultPolicyPinnedToSecondNodeReturnsAllNodes) {
    // Previously reported as {0} regardless of where the thread runs.
    auto const nodes = nodes_with_policy(MPOL_DEFAULT, {}, n1);
    ASSERT_TRUE(nodes.has_value()) << "could not apply memory policy / CPU pinning";
    EXPECT_THAT(*nodes, ElementsAre(n0, n1));
}

TEST_F(TwoNumaNodesTest, BindReturnsOnlyBoundNodes) {
    for (int node : {n0, n1}) {
        auto const nodes = nodes_with_policy(MPOL_BIND, {node});
        ASSERT_TRUE(nodes.has_value()) << "could not apply memory policy / CPU pinning";
        EXPECT_THAT(*nodes, ElementsAre(node));
    }
    auto const both = nodes_with_policy(MPOL_BIND, {n0, n1});
    ASSERT_TRUE(both.has_value()) << "could not apply memory policy / CPU pinning";
    EXPECT_THAT(*both, ElementsAre(n0, n1));
}

TEST_F(TwoNumaNodesTest, PreferredReturnsAllNodesPreferredFirst) {
    auto const first = nodes_with_policy(MPOL_PREFERRED, {n0});
    ASSERT_TRUE(first.has_value()) << "could not apply memory policy / CPU pinning";
    EXPECT_THAT(*first, ElementsAre(n0, n1));

    auto const second = nodes_with_policy(MPOL_PREFERRED, {n1});
    ASSERT_TRUE(second.has_value()) << "could not apply memory policy / CPU pinning";
    EXPECT_THAT(*second, ElementsAre(n1, n0));
}

TEST_F(TwoNumaNodesTest, InterleaveReturnsAllNodesInterleavedFirst) {
    auto const both = nodes_with_policy(MPOL_INTERLEAVE, {n0, n1});
    ASSERT_TRUE(both.has_value()) << "could not apply memory policy / CPU pinning";
    EXPECT_THAT(*both, ElementsAre(n0, n1));

    // MPOL_INTERLEAVE with a single node is accepted by the kernel.
    auto const second = nodes_with_policy(MPOL_INTERLEAVE, {n1});
    ASSERT_TRUE(second.has_value()) << "could not apply memory policy / CPU pinning";
    EXPECT_THAT(*second, ElementsAre(n1, n0));
}

// ---------------------------------------------------------------------------
// Multi-NUMA GB/GH system: GPU HBM nodes have memory but no CPUs and must never be
// reported as host memory.
// ---------------------------------------------------------------------------
class MultiNumaGBTest : public ::testing::Test {
  protected:
    void SetUp() override {
        topo = discover_topology();
        if (topo.host.size() < 2 || topo.cpuless.empty()) {
            GTEST_SKIP() << "requires a multi-NUMA system with CPU-less memory nodes";
        }
    }

    Topology topo;
};

TEST_F(MultiNumaGBTest, DefaultPolicyExcludesCpulessNodes) {
    auto const nodes = nodes_with_policy(MPOL_DEFAULT, {});
    ASSERT_TRUE(nodes.has_value()) << "could not apply memory policy / CPU pinning";
    EXPECT_THAT(*nodes, ElementsAreArray(topo.host));
}

TEST_F(MultiNumaGBTest, DefaultPolicyPinnedToEachHostNodeExcludesCpulessNodes) {
    for (int node : topo.host) {
        auto const nodes = nodes_with_policy(MPOL_DEFAULT, {}, node);
        ASSERT_TRUE(nodes.has_value()) << "could not apply memory policy / CPU pinning";
        EXPECT_THAT(*nodes, ElementsAreArray(topo.host)) << "pinned to node " << node;
    }
}

TEST_F(MultiNumaGBTest, BindReturnsOnlyBoundHostNodes) {
    for (int node : topo.host) {
        auto const nodes = nodes_with_policy(MPOL_BIND, {node});
        ASSERT_TRUE(nodes.has_value()) << "could not apply memory policy / CPU pinning";
        EXPECT_THAT(*nodes, ElementsAre(node));
    }
    auto const all = nodes_with_policy(MPOL_BIND, topo.host);
    ASSERT_TRUE(all.has_value()) << "could not apply memory policy / CPU pinning";
    EXPECT_THAT(*all, ElementsAreArray(topo.host));
}

TEST_F(MultiNumaGBTest, PreferredAndInterleaveListPolicyNodesFirstAndExcludeCpuless) {
    for (int mode : {MPOL_PREFERRED, MPOL_INTERLEAVE}) {
        for (int node : topo.host) {
            auto const nodes = nodes_with_policy(mode, {node});
            ASSERT_TRUE(nodes.has_value())
                << "could not apply memory policy / CPU pinning";
            ASSERT_EQ(nodes->size(), topo.host.size()) << "mode " << mode;
            EXPECT_EQ(nodes->front(), node) << "mode " << mode;
            EXPECT_THAT(*nodes, UnorderedElementsAreArray(topo.host));
            for (int cpuless : topo.cpuless) {
                EXPECT_THAT(*nodes, ::testing::Not(::testing::Contains(cpuless)));
            }
        }
    }
}

#else  // !RAPIDSMPF_HAVE_NUMA

TEST(GetCurrentNumaNodesTest, WithoutNumaSupportReturnsNodeZero) {
    EXPECT_THAT(get_current_numa_nodes(), ::testing::ElementsAre(0));
}

#endif  // RAPIDSMPF_HAVE_NUMA
