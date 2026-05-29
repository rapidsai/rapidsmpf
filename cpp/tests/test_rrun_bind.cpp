/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @brief GTest for the rapidsmpf::rrun::bind() API.
 *
 * GPU-resolution tests exercise the fallback chain (explicit ID ->
 * CUDA_VISIBLE_DEVICES -> error) using synthetic topology objects so they
 * can run without a real GPU.  Binding-effect tests verify that CPU
 * affinity, NUMA memory policy and UCX_NET_DEVICES are applied correctly;
 * these require a working topology_discovery and are skipped otherwise.
 *
 * Run with:  rrun -n 1 gtests/rrun_tests  (single-rank)
 *            rrun -n 4 gtests/rrun_tests  (multi-rank)
 * or directly:  gtests/rrun_tests
 */

#include <algorithm>
#include <cstdlib>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <sched.h>

#include <cucascade/memory/topology_discovery.hpp>

#include <rrun/rrun.hpp>
#include <rrun/scoped_env_var.hpp>

#include <rapidsmpf/bootstrap/utils.hpp>
#include <rapidsmpf/system_info.hpp>

using rapidsmpf::rrun::ScopedEnvVar;

/**
 * @brief Build a minimal synthetic topology containing a single GPU.
 *
 * @param gpu_id          GPU device ID to assign.
 * @param cpu_affinity    CPU affinity list string for the GPU.
 * @param memory_binding  NUMA node IDs for memory binding.
 * @param network_devices Network device names (NICs) for the GPU.
 * @return A `system_topology_info` with one GPU entry.
 */
static cucascade::memory::system_topology_info make_single_gpu_topology(
    unsigned int gpu_id,
    std::string const& cpu_affinity = "0-3",
    std::vector<int> const& memory_binding = {0},
    std::vector<std::string> const& network_devices = {"mlx5_0"}
) {
    cucascade::memory::gpu_topology_info gpu;
    gpu.id = gpu_id;
    gpu.name = "TestGPU";
    gpu.cpu_affinity_list = cpu_affinity;
    gpu.memory_binding = memory_binding;
    gpu.network_devices = network_devices;

    cucascade::memory::system_topology_info topo;
    topo.hostname = "test-host";
    topo.num_gpus = 1;
    topo.num_numa_nodes = 1;
    topo.num_network_devices = static_cast<int>(network_devices.size());
    topo.gpus.push_back(gpu);
    return topo;
}

/**
 * @brief Tests for GPU-ID resolution logic (no real GPU or topology needed).
 *
 * All tests disable actual resource binding so that they only exercise the
 * resolution path without modifying CPU affinity, NUMA policy, or environment
 * variables of the calling process.
 */
class RrunBindResolution : public ::testing::Test {
  protected:
    static constexpr rapidsmpf::rrun::bind_options noop_opts{
        .cpu = false, .memory = false, .network = false
    };
};

TEST_F(RrunBindResolution, ExplicitGpuIdUsedOverEnvVar) {
    ScopedEnvVar env("CUDA_VISIBLE_DEVICES", "99");
    auto topo = make_single_gpu_topology(7);

    EXPECT_NO_THROW(rapidsmpf::rrun::bind(topo, 7u, noop_opts));
}

TEST_F(RrunBindResolution, FallbackToCudaVisibleDevices) {
    ScopedEnvVar env("CUDA_VISIBLE_DEVICES", "5");
    auto topo = make_single_gpu_topology(5);

    EXPECT_NO_THROW(rapidsmpf::rrun::bind(topo, std::nullopt, noop_opts));
}

TEST_F(RrunBindResolution, FallbackUsesFirstEntry) {
    ScopedEnvVar env("CUDA_VISIBLE_DEVICES", "3,4,5");
    auto topo = make_single_gpu_topology(3);

    EXPECT_NO_THROW(rapidsmpf::rrun::bind(topo, std::nullopt, noop_opts));
}

TEST_F(RrunBindResolution, ThrowsWhenNoGpuIdAndNoEnvVar) {
    ScopedEnvVar env("CUDA_VISIBLE_DEVICES", nullptr);

    auto topo = make_single_gpu_topology(0);
    EXPECT_THROW(
        rapidsmpf::rrun::bind(topo, std::nullopt, noop_opts), std::runtime_error
    );
}

TEST_F(RrunBindResolution, ThrowsWhenEnvVarEmpty) {
    ScopedEnvVar env("CUDA_VISIBLE_DEVICES", "");

    auto topo = make_single_gpu_topology(0);
    EXPECT_THROW(
        rapidsmpf::rrun::bind(topo, std::nullopt, noop_opts), std::runtime_error
    );
}

TEST_F(RrunBindResolution, ThrowsWhenEnvVarIsUuid) {
    ScopedEnvVar env("CUDA_VISIBLE_DEVICES", "GPU-abcdef12-3456-7890");

    auto topo = make_single_gpu_topology(0);
    EXPECT_THROW(
        rapidsmpf::rrun::bind(topo, std::nullopt, noop_opts), std::runtime_error
    );
}

TEST_F(RrunBindResolution, ThrowsWhenEnvVarIsNegative) {
    ScopedEnvVar env("CUDA_VISIBLE_DEVICES", "-1");

    auto topo = make_single_gpu_topology(0);
    EXPECT_THROW(
        rapidsmpf::rrun::bind(topo, std::nullopt, noop_opts), std::runtime_error
    );
}

TEST_F(RrunBindResolution, GpuNotInTopologyThrows) {
    auto topo = make_single_gpu_topology(0);

    EXPECT_THROW(rapidsmpf::rrun::bind(topo, 42u, noop_opts), std::runtime_error);
}

/**
 * @brief Tests for binding side-effects using synthetic topology.
 *
 * Saves and restores CPU affinity and `UCX_NET_DEVICES` around each test so
 * that tests do not permanently alter the process state.
 */
class RrunBindEffect : public ::testing::Test {
  protected:
    void SetUp() override {
        CPU_ZERO(&saved_cpuset_);
        sched_getaffinity(0, sizeof(saved_cpuset_), &saved_cpuset_);

        char const* val = std::getenv("UCX_NET_DEVICES");
        if (val != nullptr) {
            had_ucx_net_ = true;
            old_ucx_net_ = val;
        }
    }

    void TearDown() override {
        sched_setaffinity(0, sizeof(saved_cpuset_), &saved_cpuset_);

        if (had_ucx_net_) {
            setenv("UCX_NET_DEVICES", old_ucx_net_.c_str(), 1);
        } else {
            unsetenv("UCX_NET_DEVICES");
        }
    }

    cpu_set_t saved_cpuset_{};
    bool had_ucx_net_{false};
    std::string old_ucx_net_;
};

TEST_F(RrunBindEffect, NetworkBindingSetsEnvVar) {
    auto topo = make_single_gpu_topology(0, "0-3", {0}, {"mlx5_2", "mlx5_3"});

    rapidsmpf::rrun::bind(topo, 0u, {.cpu = false, .memory = false, .network = true});

    char const* val = std::getenv("UCX_NET_DEVICES");
    ASSERT_NE(val, nullptr);
    EXPECT_EQ(std::string(val), "mlx5_2,mlx5_3");
}

TEST_F(RrunBindEffect, NetworkBindingSkippedWhenDisabled) {
    unsetenv("UCX_NET_DEVICES");

    auto topo = make_single_gpu_topology(0, "0-3", {0}, {"mlx5_0"});

    rapidsmpf::rrun::bind(topo, 0u, {.cpu = false, .memory = false, .network = false});

    EXPECT_EQ(std::getenv("UCX_NET_DEVICES"), nullptr);
}

TEST_F(RrunBindEffect, CpuBindingAppliesAffinity) {
    auto topo = make_single_gpu_topology(0, "0-3", {}, {});

    rapidsmpf::rrun::bind(topo, 0u, {.cpu = true, .memory = false, .network = false});

    std::string affinity = rapidsmpf::bootstrap::get_current_cpu_affinity();
    EXPECT_TRUE(rapidsmpf::bootstrap::compare_cpu_affinity(affinity, "0-3"))
        << "Expected CPU affinity 0-3, got: " << affinity;
}

TEST_F(RrunBindEffect, CpuBindingSkippedWhenDisabled) {
    std::string before = rapidsmpf::bootstrap::get_current_cpu_affinity();

    auto topo = make_single_gpu_topology(0, "0-1", {}, {});

    rapidsmpf::rrun::bind(topo, 0u, {.cpu = false, .memory = false, .network = false});

    std::string after = rapidsmpf::bootstrap::get_current_cpu_affinity();
    EXPECT_EQ(before, after);
}

/**
 * @brief Live-system tests that exercise bind() against real topology and
 * verify the resulting CPU affinity, NUMA binding and UCX_NET_DEVICES.
 *
 * When running under rrun the fixture binds to the rank's own GPU (from
 * `CUDA_VISIBLE_DEVICES`), otherwise it falls back to the first GPU in
 * the discovered topology.  CPU affinity and `UCX_NET_DEVICES` are saved
 * and restored around each test.
 *
 * Skips automatically when topology discovery fails or no GPUs are found.
 */
class RrunBindLive : public ::testing::Test {
  protected:
    void SetUp() override {
        // Read the physical GPU ID from CUDA_VISIBLE_DEVICES before clearing
        // it; topology discovery must see all GPUs, not just the one rrun
        // narrowed the env var to.
        int env_gpu = rapidsmpf::bootstrap::get_gpu_id();

        {
            ScopedEnvVar wide_view("CUDA_VISIBLE_DEVICES", nullptr);
            if (!discovery_.discover()) {
                GTEST_SKIP() << "Topology discovery unavailable";
            }
        }

        auto const& topo = discovery_.get_topology();
        if (topo.gpus.empty()) {
            GTEST_SKIP() << "No GPUs found in topology";
        }

        if (env_gpu >= 0) {
            gpu_id_ = static_cast<unsigned int>(env_gpu);
        } else {
            gpu_id_ = topo.gpus.front().id;
        }

        auto it = std::find_if(topo.gpus.begin(), topo.gpus.end(), [this](auto const& g) {
            return g.id == gpu_id_;
        });
        if (it == topo.gpus.end()) {
            GTEST_SKIP() << "GPU " << gpu_id_ << " not found in topology";
        }
        expected_gpu_info_ = *it;

        CPU_ZERO(&saved_cpuset_);
        sched_getaffinity(0, sizeof(saved_cpuset_), &saved_cpuset_);

        char const* val = std::getenv("UCX_NET_DEVICES");
        if (val != nullptr) {
            had_ucx_net_ = true;
            old_ucx_net_ = val;
        }
    }

    void TearDown() override {
        sched_setaffinity(0, sizeof(saved_cpuset_), &saved_cpuset_);

        if (had_ucx_net_) {
            setenv("UCX_NET_DEVICES", old_ucx_net_.c_str(), 1);
        } else {
            unsetenv("UCX_NET_DEVICES");
        }
    }

    cucascade::memory::topology_discovery discovery_;
    unsigned int gpu_id_{0};
    cucascade::memory::gpu_topology_info expected_gpu_info_;
    cpu_set_t saved_cpuset_{};
    bool had_ucx_net_{false};
    std::string old_ucx_net_;
};

TEST_F(RrunBindLive, CpuAffinity) {
    rapidsmpf::rrun::bind(
        discovery_.get_topology(),
        gpu_id_,
        {.cpu = true, .memory = false, .network = false}
    );

    if (expected_gpu_info_.cpu_affinity_list.empty()) {
        GTEST_SKIP() << "No CPU affinity expected for GPU " << gpu_id_;
    }

    std::string actual = rapidsmpf::bootstrap::get_current_cpu_affinity();
    EXPECT_TRUE(
        rapidsmpf::bootstrap::compare_cpu_affinity(
            actual, expected_gpu_info_.cpu_affinity_list
        )
    ) << "CPU affinity mismatch for GPU "
      << gpu_id_ << "\n"
      << "  Expected: " << expected_gpu_info_.cpu_affinity_list << "\n"
      << "  Actual:   " << actual;
}

TEST_F(RrunBindLive, NumaBinding) {
    rapidsmpf::rrun::bind(
        discovery_.get_topology(),
        gpu_id_,
        {.cpu = false, .memory = true, .network = false}
    );

    if (expected_gpu_info_.memory_binding.empty()) {
        GTEST_SKIP() << "No NUMA binding expected for GPU " << gpu_id_;
    }

    std::vector<int> actual_nodes = rapidsmpf::get_current_numa_nodes();
    ASSERT_FALSE(actual_nodes.empty())
        << "No NUMA nodes detected after binding for GPU " << gpu_id_;

    bool found = std::any_of(actual_nodes.begin(), actual_nodes.end(), [this](int node) {
        return std::find(
                   expected_gpu_info_.memory_binding.begin(),
                   expected_gpu_info_.memory_binding.end(),
                   node
               )
               != expected_gpu_info_.memory_binding.end();
    });

    auto format_nodes = [](std::vector<int> const& v) {
        std::ostringstream oss;
        oss << "[";
        for (std::size_t i = 0; i < v.size(); ++i) {
            if (i > 0)
                oss << ",";
            oss << v[i];
        }
        oss << "]";
        return oss.str();
    };

    EXPECT_TRUE(found) << "NUMA binding mismatch for GPU " << gpu_id_ << "\n"
                       << "  Expected: "
                       << format_nodes(expected_gpu_info_.memory_binding) << "\n"
                       << "  Actual:   " << format_nodes(actual_nodes);
}

TEST_F(RrunBindLive, UcxNetDevices) {
    rapidsmpf::rrun::bind(
        discovery_.get_topology(),
        gpu_id_,
        {.cpu = false, .memory = false, .network = true}
    );

    if (expected_gpu_info_.network_devices.empty()) {
        GTEST_SKIP() << "No network devices expected for GPU " << gpu_id_;
    }

    std::string expected;
    for (std::size_t i = 0; i < expected_gpu_info_.network_devices.size(); ++i) {
        if (i > 0)
            expected += ",";
        expected += expected_gpu_info_.network_devices[i];
    }

    std::string actual = rapidsmpf::bootstrap::get_ucx_net_devices();
    EXPECT_TRUE(rapidsmpf::bootstrap::compare_device_lists(actual, expected))
        << "UCX_NET_DEVICES mismatch for GPU " << gpu_id_ << "\n"
        << "  Expected: " << expected << "\n"
        << "  Actual:   " << actual;
}

TEST_F(RrunBindLive, CudaVisibleDevicesFallback) {
    ScopedEnvVar env("CUDA_VISIBLE_DEVICES", std::to_string(gpu_id_).c_str());

    rapidsmpf::rrun::bind(
        discovery_.get_topology(),
        std::nullopt,
        {.cpu = true, .memory = true, .network = true}
    );

    if (!expected_gpu_info_.cpu_affinity_list.empty()) {
        std::string actual = rapidsmpf::bootstrap::get_current_cpu_affinity();
        EXPECT_TRUE(
            rapidsmpf::bootstrap::compare_cpu_affinity(
                actual, expected_gpu_info_.cpu_affinity_list
            )
        ) << "CPU affinity mismatch (CUDA_VISIBLE_DEVICES fallback)";
    }
}

TEST_F(RrunBindLive, AutoDiscoveryOverload) {
    rapidsmpf::rrun::bind(gpu_id_, {.cpu = true, .memory = true, .network = true});

    if (!expected_gpu_info_.cpu_affinity_list.empty()) {
        std::string actual = rapidsmpf::bootstrap::get_current_cpu_affinity();
        EXPECT_TRUE(
            rapidsmpf::bootstrap::compare_cpu_affinity(
                actual, expected_gpu_info_.cpu_affinity_list
            )
        ) << "CPU affinity mismatch (auto-discovery overload)";
    }
}

class RrunValidateBinding : public ::testing::Test {};

TEST_F(RrunValidateBinding, AllPassWhenMatching) {
    rapidsmpf::rrun::resource_binding actual;
    actual.cpu_affinity = "0-3";
    actual.numa_nodes = {0};
    actual.ucx_net_devices = "mlx5_0";

    rapidsmpf::rrun::expected_binding expected;
    expected.cpu_affinity = "0-3";
    expected.memory_binding = {0};
    expected.network_devices = {"mlx5_0"};

    auto result = rapidsmpf::rrun::validate_binding(actual, expected);
    EXPECT_TRUE(result.cpu_ok);
    EXPECT_TRUE(result.numa_ok);
    EXPECT_TRUE(result.ucx_ok);
    EXPECT_TRUE(result.all_passed());
}

TEST_F(RrunValidateBinding, CpuMismatchDetected) {
    rapidsmpf::rrun::resource_binding actual;
    actual.cpu_affinity = "4-7";

    rapidsmpf::rrun::expected_binding expected;
    expected.cpu_affinity = "0-3";

    auto result = rapidsmpf::rrun::validate_binding(actual, expected);
    EXPECT_FALSE(result.cpu_ok);
    EXPECT_FALSE(result.all_passed());
}

TEST_F(RrunValidateBinding, NumaMismatchDetected) {
    rapidsmpf::rrun::resource_binding actual;
    actual.numa_nodes = {1};

    rapidsmpf::rrun::expected_binding expected;
    expected.memory_binding = {0};

    auto result = rapidsmpf::rrun::validate_binding(actual, expected);
    EXPECT_FALSE(result.numa_ok);
    EXPECT_FALSE(result.all_passed());
}

TEST_F(RrunValidateBinding, NumaPassesWhenAnyNodeMatches) {
    rapidsmpf::rrun::resource_binding actual;
    actual.numa_nodes = {1, 0};

    rapidsmpf::rrun::expected_binding expected;
    expected.memory_binding = {0};

    auto result = rapidsmpf::rrun::validate_binding(actual, expected);
    EXPECT_TRUE(result.numa_ok);
}

TEST_F(RrunValidateBinding, NumaEmptyExpectedIsPass) {
    rapidsmpf::rrun::resource_binding actual;
    actual.numa_nodes = {3};

    rapidsmpf::rrun::expected_binding expected;

    auto result = rapidsmpf::rrun::validate_binding(actual, expected);
    EXPECT_TRUE(result.numa_ok);
    EXPECT_TRUE(result.all_passed());
}

TEST_F(RrunValidateBinding, UcxMismatchDetected) {
    rapidsmpf::rrun::resource_binding actual;
    actual.ucx_net_devices = "mlx5_1";

    rapidsmpf::rrun::expected_binding expected;
    expected.network_devices = {"mlx5_0"};

    auto result = rapidsmpf::rrun::validate_binding(actual, expected);
    EXPECT_FALSE(result.ucx_ok);
    EXPECT_EQ(result.expected_ucx_devices, "mlx5_0");
    EXPECT_FALSE(result.all_passed());
}

TEST_F(RrunValidateBinding, UcxOrderIndependent) {
    rapidsmpf::rrun::resource_binding actual;
    actual.ucx_net_devices = "mlx5_1,mlx5_0";

    rapidsmpf::rrun::expected_binding expected;
    expected.network_devices = {"mlx5_0", "mlx5_1"};

    auto result = rapidsmpf::rrun::validate_binding(actual, expected);
    EXPECT_TRUE(result.ucx_ok);
}

TEST_F(RrunValidateBinding, EmptyExpectedIsAllPass) {
    rapidsmpf::rrun::resource_binding actual;
    actual.cpu_affinity = "0-7";
    actual.numa_nodes = {0};
    actual.ucx_net_devices = "mlx5_0";

    rapidsmpf::rrun::expected_binding expected;

    auto result = rapidsmpf::rrun::validate_binding(actual, expected);
    EXPECT_TRUE(result.all_passed());
}

class RrunGetExpectedBinding : public ::testing::Test {};

TEST_F(RrunGetExpectedBinding, ReturnsBindingForKnownGpu) {
    auto topo = make_single_gpu_topology(3, "8-15", {1}, {"mlx5_2"});

    auto result = rapidsmpf::rrun::get_expected_binding(topo, 3);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->cpu_affinity, "8-15");
    ASSERT_EQ(result->memory_binding.size(), 1u);
    EXPECT_EQ(result->memory_binding[0], 1);
    ASSERT_EQ(result->network_devices.size(), 1u);
    EXPECT_EQ(result->network_devices[0], "mlx5_2");
}

TEST_F(RrunGetExpectedBinding, ReturnsNulloptForUnknownGpu) {
    auto topo = make_single_gpu_topology(0);

    auto result = rapidsmpf::rrun::get_expected_binding(topo, 99);
    EXPECT_FALSE(result.has_value());
}

TEST_F(RrunGetExpectedBinding, ReturnsEmptyFieldsWhenTopologyLacks) {
    cucascade::memory::gpu_topology_info gpu;
    gpu.id = 0;
    gpu.name = "TestGPU";

    cucascade::memory::system_topology_info topo;
    topo.hostname = "test-host";
    topo.num_gpus = 1;
    topo.num_numa_nodes = 0;
    topo.num_network_devices = 0;
    topo.gpus.push_back(gpu);

    auto result = rapidsmpf::rrun::get_expected_binding(topo, 0);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->cpu_affinity.empty());
    EXPECT_TRUE(result->memory_binding.empty());
    EXPECT_TRUE(result->network_devices.empty());
}

class RrunCheckBindingLive : public ::testing::Test {
  protected:
    void SetUp() override {
        int env_gpu = rapidsmpf::bootstrap::get_gpu_id();
        if (env_gpu < 0) {
            GTEST_SKIP() << "No GPU ID available (CUDA_VISIBLE_DEVICES not set)";
        }
        gpu_id_ = env_gpu;
    }

    int gpu_id_{-1};
};

TEST_F(RrunCheckBindingLive, ReturnsPopulatedBinding) {
    auto binding = rapidsmpf::rrun::check_binding(gpu_id_);

    EXPECT_EQ(binding.gpu_id, gpu_id_);
    EXPECT_FALSE(binding.cpu_affinity.empty());
    EXPECT_FALSE(binding.numa_nodes.empty());
}

TEST_F(RrunCheckBindingLive, GpuIdHintIsStored) {
    auto binding = rapidsmpf::rrun::check_binding(42);
    EXPECT_EQ(binding.gpu_id, 42);
}

TEST_F(RrunCheckBindingLive, NegativeHintFallsToCvd) {
    ScopedEnvVar env("CUDA_VISIBLE_DEVICES", std::to_string(gpu_id_).c_str());
    auto binding = rapidsmpf::rrun::check_binding(-1);
    EXPECT_EQ(binding.gpu_id, gpu_id_);
}

TEST_F(RrunCheckBindingLive, BindThenCheckMatchesExpected) {
    cucascade::memory::topology_discovery discovery;
    {
        ScopedEnvVar wide_view("CUDA_VISIBLE_DEVICES", nullptr);
        if (!discovery.discover()) {
            GTEST_SKIP() << "Topology discovery unavailable";
        }
    }

    auto expected_opt =
        rapidsmpf::rrun::get_expected_binding(discovery.get_topology(), gpu_id_);
    if (!expected_opt) {
        GTEST_SKIP() << "GPU " << gpu_id_ << " not found in topology";
    }

    rapidsmpf::rrun::bind(
        discovery.get_topology(),
        static_cast<unsigned int>(gpu_id_),
        {.cpu = true, .memory = true, .network = true, .verify = false}
    );

    auto actual = rapidsmpf::rrun::check_binding(gpu_id_);
    auto validation = rapidsmpf::rrun::validate_binding(actual, *expected_opt);

    EXPECT_TRUE(validation.all_passed())
        << "cpu_ok=" << validation.cpu_ok << " numa_ok=" << validation.numa_ok
        << " ucx_ok=" << validation.ucx_ok;
}
