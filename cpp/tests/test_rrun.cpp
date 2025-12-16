/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @brief GTest for validating topology-based binding in rrun.
 *
 * This test validates that when running under rrun, the CPU affinity, NUMA memory
 * binding, and UCX_NET_DEVICES match what TopologyDiscovery reports for the assigned GPU.
 *
 * These tests must be run with rrun, e.g.:
 *   rrun -n 1 gtests/single_tests --gtest_filter="*TopologyBinding*"
 */

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rapidsmpf/bootstrap/utils.hpp>
#include <rapidsmpf/system_info.hpp>
#include <rapidsmpf/topology_discovery.hpp>

class TopologyBindingTest : public ::testing::Test {
  protected:
    void SetUp() override {
        if (!rapidsmpf::bootstrap::is_running_with_rrun()) {
            GTEST_SKIP() << "Test must be run with rrun (RAPIDSMPF_RANK not set)";
        }

        if (!discovery_.discover()) {
            GTEST_SKIP() << "Failed to discover topology";
        }

        gpu_id_ = rapidsmpf::bootstrap::get_gpu_id();
        if (gpu_id_ < 0) {
            GTEST_SKIP() << "Could not determine GPU ID from CUDA_VISIBLE_DEVICES.";
        }

        auto const& topology = discovery_.get_topology();
        auto it = std::find_if(
            topology.gpus.begin(),
            topology.gpus.end(),
            [this](rapidsmpf::GpuTopologyInfo const& gpu) {
                return static_cast<int>(gpu.id) == gpu_id_;
            }
        );

        if (it == topology.gpus.end()) {
            GTEST_SKIP() << "GPU ID " << gpu_id_ << " not found in topology";
        }

        expected_gpu_info_ = *it;
    }

    rapidsmpf::TopologyDiscovery discovery_;
    int gpu_id_{-1};
    rapidsmpf::GpuTopologyInfo expected_gpu_info_;
};

TEST_F(TopologyBindingTest, CpuAffinity) {
    std::string actual_cpu_affinity = rapidsmpf::bootstrap::get_current_cpu_affinity();
    std::string expected_cpu_affinity = expected_gpu_info_.cpu_affinity_list;

    if (expected_cpu_affinity.empty()) {
        GTEST_SKIP() << "No CPU affinity expected for GPU " << gpu_id_;
    }

    EXPECT_TRUE(
        rapidsmpf::bootstrap::compare_cpu_affinity(
            actual_cpu_affinity, expected_cpu_affinity
        )
    ) << "CPU affinity mismatch for GPU "
      << gpu_id_ << "\n"
      << "  Expected: " << expected_cpu_affinity << "\n"
      << "  Actual:   " << actual_cpu_affinity;
}

TEST_F(TopologyBindingTest, NumaBinding) {
    std::vector<int> actual_numa_nodes = rapidsmpf::get_current_numa_nodes();
    std::vector<int> expected_memory_binding = expected_gpu_info_.memory_binding;

    if (expected_memory_binding.empty()) {
        GTEST_SKIP() << "No NUMA binding expected for GPU " << gpu_id_;
    }

    if (actual_numa_nodes.empty()) {
        std::ostringstream oss;
        oss << "No NUMA nodes detected, but expected binding to: [";
        for (size_t i = 0; i < expected_memory_binding.size(); ++i) {
            if (i > 0)
                oss << ",";
            oss << expected_memory_binding[i];
        }
        oss << "]";
        FAIL() << oss.str();
    }

    // Check if any actual NUMA node is in expected list
    bool found = false;
    for (int actual_node : actual_numa_nodes) {
        if (std::find(
                expected_memory_binding.begin(),
                expected_memory_binding.end(),
                actual_node
            )
            != expected_memory_binding.end())
        {
            found = true;
            break;
        }
    }

    EXPECT_TRUE(found) << "NUMA binding mismatch for GPU " << gpu_id_ << "\n"
                       << "  Expected: [";
    for (size_t i = 0; i < expected_memory_binding.size(); ++i) {
        if (i > 0)
            std::cout << ",";
        std::cout << expected_memory_binding[i];
    }
    std::cout << "]\n"
              << "  Actual:   [";
    for (size_t i = 0; i < actual_numa_nodes.size(); ++i) {
        if (i > 0)
            std::cout << ",";
        std::cout << actual_numa_nodes[i];
    }
    std::cout << "]";
}

TEST_F(TopologyBindingTest, UcxNetDevices) {
    std::string actual_ucx_net_devices = rapidsmpf::bootstrap::get_ucx_net_devices();
    std::vector<std::string> expected_network_devices =
        expected_gpu_info_.network_devices;

    if (expected_network_devices.empty()) {
        GTEST_SKIP() << "No network devices expected for GPU " << gpu_id_;
    }

    // Convert expected network devices to comma-separated string
    std::string expected_ucx_devices;
    for (size_t i = 0; i < expected_network_devices.size(); ++i) {
        if (i > 0)
            expected_ucx_devices += ",";
        expected_ucx_devices += expected_network_devices[i];
    }

    EXPECT_TRUE(
        rapidsmpf::bootstrap::compare_device_lists(
            actual_ucx_net_devices, expected_ucx_devices
        )
    ) << "UCX_NET_DEVICES mismatch for GPU "
      << gpu_id_ << "\n"
      << "  Expected: " << expected_ucx_devices << "\n"
      << "  Actual:   " << actual_ucx_net_devices;
}

TEST_F(TopologyBindingTest, AllBindings) {
    std::string actual_cpu_affinity = rapidsmpf::bootstrap::get_current_cpu_affinity();
    std::vector<int> actual_numa_nodes = rapidsmpf::get_current_numa_nodes();
    std::string actual_ucx_net_devices = rapidsmpf::bootstrap::get_ucx_net_devices();

    // Check CPU affinity
    if (!expected_gpu_info_.cpu_affinity_list.empty()) {
        bool cpu_ok = rapidsmpf::bootstrap::compare_cpu_affinity(
            actual_cpu_affinity, expected_gpu_info_.cpu_affinity_list
        );
        EXPECT_TRUE(cpu_ok) << "CPU affinity mismatch";
    }

    // Check NUMA binding
    if (!expected_gpu_info_.memory_binding.empty()) {
        bool numa_ok = false;
        if (!actual_numa_nodes.empty()) {
            for (int actual_node : actual_numa_nodes) {
                if (std::find(
                        expected_gpu_info_.memory_binding.begin(),
                        expected_gpu_info_.memory_binding.end(),
                        actual_node
                    )
                    != expected_gpu_info_.memory_binding.end())
                {
                    numa_ok = true;
                    break;
                }
            }
        }
        EXPECT_TRUE(numa_ok) << "NUMA binding mismatch";
    }

    // Check UCX network devices
    if (!expected_gpu_info_.network_devices.empty()) {
        std::string expected_ucx_devices;
        for (size_t i = 0; i < expected_gpu_info_.network_devices.size(); ++i) {
            if (i > 0)
                expected_ucx_devices += ",";
            expected_ucx_devices += expected_gpu_info_.network_devices[i];
        }
        bool ucx_ok = rapidsmpf::bootstrap::compare_device_lists(
            actual_ucx_net_devices, expected_ucx_devices
        );
        EXPECT_TRUE(ucx_ok) << "UCX_NET_DEVICES mismatch";
    }
}
