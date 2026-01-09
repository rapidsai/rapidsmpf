/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @brief Test program to validate topology-based binding in rrun.
 *
 * This program queries its CPU affinity, NUMA memory binding, and UCX_NET_DEVICES
 * environment variable, then reports them and optionally validates them against
 * expected values from topology discovery or a JSON file.
 *
 * Usage:
 *   check_resource_binding [--json <topology.json>] [--gpu-id <id>]
 *
 * If --json is provided, reads expected values from the JSON file.
 * If --gpu-id is provided, uses cucascade::memory::topology_discovery API to get expected
 * values. Otherwise, just reports current configuration.
 */

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include <rapidsmpf/bootstrap/utils.hpp>
#include <rapidsmpf/system_info.hpp>

#include <memory/topology_discovery.hpp>

namespace {

/**
 * @brief Simple JSON string value extractor using a regex.
 *
 * Matches patterns of the form:
 *     "key" : "value"
 *
 * This is not a full JSON parser and assumes reasonably simple,
 * flat key–value pairs without escaped quotes.
 */
std::string extract_json_string_value(std::string const& json, std::string const& key) {
    // Regex: "key" <optional spaces> : <optional spaces> "value"
    std::string pattern = "\\\"" + key + "\\\"\\s*:\\s*\\\"([^\\\"]*)\\\"";
    std::regex re(pattern);
    std::smatch match;
    if (std::regex_search(json, match, re)) {
        // match[1] is the captured value
        return match[1].str();
    }
    return "";
}

/**
 * @brief Extract GPU info from JSON for a specific GPU ID.
 */
bool extract_gpu_info_from_json(
    std::string const& json_file,
    int gpu_id,
    std::string& cpu_affinity,
    std::vector<int>& memory_binding,
    std::vector<std::string>& network_devices
) {
    std::ifstream file(json_file);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open JSON file: " << json_file << std::endl;
        return false;
    }

    std::string json_content(
        (std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>()
    );

    // Find the GPU with matching ID
    std::string gpu_id_str = "\"id\": " + std::to_string(gpu_id);
    size_t gpu_pos = json_content.find(gpu_id_str);
    if (gpu_pos == std::string::npos) {
        std::cerr << "Error: GPU ID " << gpu_id << " not found in JSON" << std::endl;
        return false;
    }

    // Find the start of this GPU object
    size_t gpu_start = json_content.rfind('{', gpu_pos);
    if (gpu_start == std::string::npos) {
        return false;
    }

    // Find the end of this GPU object by matching braces
    // (accounting for nested objects like cpu_affinity)
    int brace_count = 0;
    size_t gpu_end = gpu_start;
    bool in_string = false;
    for (size_t i = gpu_start; i < json_content.length(); ++i) {
        char c = json_content[i];
        if (c == '"') {
            // Check if this quote is escaped
            bool escaped = false;
            if (i > 0) {
                // Count backslashes - odd number means escaped
                size_t backslash_count = 0;
                for (size_t j = i - 1; j >= gpu_start && json_content[j] == '\\'; --j) {
                    backslash_count++;
                }
                escaped = (backslash_count % 2 == 1);
            }
            if (!escaped) {
                in_string = !in_string;
            }
        } else if (!in_string) {
            if (c == '{') {
                brace_count++;
            } else if (c == '}') {
                brace_count--;
                if (brace_count == 0) {
                    gpu_end = i;
                    break;
                }
            }
        }
    }
    if (brace_count != 0) {
        return false;
    }

    std::string gpu_json = json_content.substr(gpu_start, gpu_end - gpu_start + 1);

    // Extract cpu_affinity.cpulist
    cpu_affinity = extract_json_string_value(gpu_json, "cpulist");

    // Extract memory_binding array (simplified - just get first value)
    size_t membind_pos = gpu_json.find("\"memory_binding\"");
    if (membind_pos != std::string::npos) {
        size_t array_start = gpu_json.find('[', membind_pos);
        if (array_start != std::string::npos) {
            size_t array_end = gpu_json.find(']', array_start);
            if (array_end != std::string::npos) {
                std::string array_str =
                    gpu_json.substr(array_start + 1, array_end - array_start - 1);
                std::istringstream iss(array_str);
                std::string item;
                while (std::getline(iss, item, ',')) {
                    try {
                        memory_binding.push_back(std::stoi(item));
                    } catch (...) {
                        // Ignore parse errors
                    }
                }
            }
        }
    }

    // Extract network_devices array
    size_t netdev_pos = gpu_json.find("\"network_devices\"");
    if (netdev_pos != std::string::npos) {
        size_t array_start = gpu_json.find('[', netdev_pos);
        if (array_start != std::string::npos) {
            size_t array_end = gpu_json.find(']', array_start);
            if (array_end != std::string::npos) {
                std::string array_str =
                    gpu_json.substr(array_start + 1, array_end - array_start - 1);
                // Trim whitespace
                while (!array_str.empty() && array_str[0] == ' ') {
                    array_str.erase(0, 1);
                }
                while (!array_str.empty() && array_str.back() == ' ') {
                    array_str.pop_back();
                }
                if (!array_str.empty()) {
                    std::istringstream iss(array_str);
                    std::string item;
                    while (std::getline(iss, item, ',')) {
                        // Remove quotes and whitespace
                        while (!item.empty() && item[0] == ' ') {
                            item.erase(0, 1);
                        }
                        while (!item.empty() && item.back() == ' ') {
                            item.pop_back();
                        }
                        // Remove surrounding quotes if present
                        if (item.size() >= 2 && item[0] == '"' && item.back() == '"') {
                            item = item.substr(1, item.size() - 2);
                        }
                        if (!item.empty()) {
                            network_devices.push_back(item);
                        }
                    }
                }
            }
        }
    }

    return true;
}

}  // namespace

/**
 * @brief Print or validate binding configuration.
 *
 * See top of this file for more details.
 */
int main(int argc, char* argv[]) {
    std::string json_file;
    int gpu_id = -1;
    bool validate = false;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--json" && i + 1 < argc) {
            json_file = argv[++i];
            validate = true;
        } else if (arg == "--gpu-id" && i + 1 < argc) {
            gpu_id = std::stoi(argv[++i]);
            validate = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0]
                      << " [--json <topology.json>] [--gpu-id <id>]\n"
                      << "\n"
                      << "Options:\n"
                      << "  --json <file>    Read expected values from JSON file\n"
                      << "  --gpu-id <id>    Use topology discovery API for GPU ID\n"
                      << "  --help, -h       Show this help message\n"
                      << "\n"
                      << "If neither --json nor --gpu-id is provided, "
                      << "just reports current configuration.\n"
                      << "\n"
                      << "Examples usage to validate resource binding on GPU 3 only:\n"
                      << "  $ topology_discovery > topology.json\n"
                      << "  $ rrun -n 1 -g 3 check_resource_binding--json topology.json\n"
                      << "    === Topology Binding Test ===\n"
                      << "    Rank: 0\n"
                      << "    GPU ID: 3\n"
                      << "    CPU Affinity: 0-19,40-59\n"
                      << "    NUMA Nodes: 0\n"
                      << "    UCX_NET_DEVICES: mlx5_1\n"
                      << "    \n"
                      << "    === Validation ===\n"
                      << "    CPU Affinity: PASS\n"
                      << "    NUMA Binding: PASS\n"
                      << "    UCX_NET_DEVICES: PASS\n"
                      << "    \n"
                      << "    === Result ===\n"
                      << "    All checks PASSED\n"
                      << "\n"
                      << std::endl;
            return 0;
        }
    }

    // Query current configuration
    std::string actual_cpu_affinity = rapidsmpf::bootstrap::get_current_cpu_affinity();
    std::vector<int> actual_numa_nodes = rapidsmpf::get_current_numa_nodes();
    std::string actual_ucx_net_devices = rapidsmpf::bootstrap::get_ucx_net_devices();

    int rank = rapidsmpf::bootstrap::get_rank();

    if (gpu_id < 0) {
        gpu_id = rapidsmpf::bootstrap::get_gpu_id();
    }

    // Output current configuration
    std::cout << "=== Topology Binding Test ===" << std::endl;
    if (rank >= 0) {
        std::cout << "Rank: " << rank << std::endl;
    }
    if (gpu_id >= 0) {
        std::cout << "GPU ID: " << gpu_id << std::endl;
    }
    std::cout << "CPU Affinity: "
              << (actual_cpu_affinity.empty() ? "(none)" : actual_cpu_affinity)
              << std::endl;
    std::cout << "NUMA Nodes: ";
    if (actual_numa_nodes.empty()) {
        std::cout << "(none)";
    } else {
        for (size_t i = 0; i < actual_numa_nodes.size(); ++i) {
            if (i > 0)
                std::cout << ",";
            std::cout << actual_numa_nodes[i];
        }
    }
    std::cout << std::endl;
    std::cout << "UCX_NET_DEVICES: "
              << (actual_ucx_net_devices.empty() ? "(not set)" : actual_ucx_net_devices)
              << std::endl;

    if (!validate) {
        return 0;
    }

    // Get expected values
    std::string expected_cpu_affinity;
    std::vector<int> expected_memory_binding;
    std::vector<std::string> expected_network_devices;

    if (!json_file.empty()) {
        if (!extract_gpu_info_from_json(
                json_file,
                gpu_id,
                expected_cpu_affinity,
                expected_memory_binding,
                expected_network_devices
            ))
        {
            return 1;
        }
    } else if (gpu_id >= 0) {
        cucascade::memory::topology_discovery discovery;
        if (!discovery.discover()) {
            std::cerr << "Error: Failed to discover topology" << std::endl;
            return 1;
        }

        auto const& topology = discovery.get_topology();
        auto it = std::find_if(
            topology.gpus.begin(),
            topology.gpus.end(),
            [gpu_id](cucascade::memory::gpu_topology_info const& gpu) {
                return static_cast<int>(gpu.id) == gpu_id;
            }
        );

        if (it == topology.gpus.end()) {
            std::cerr << "Error: GPU ID " << gpu_id << " not found in topology"
                      << std::endl;
            return 1;
        }

        expected_cpu_affinity = it->cpu_affinity_list;
        expected_memory_binding = it->memory_binding;
        expected_network_devices = it->network_devices;
    } else {
        std::cerr << "Error: Must provide --json or --gpu-id for validation" << std::endl;
        return 1;
    }

    std::cout << "\n=== Validation ===" << std::endl;
    bool all_passed = true;

    // Check CPU affinity
    bool cpu_ok = rapidsmpf::bootstrap::compare_cpu_affinity(
        actual_cpu_affinity, expected_cpu_affinity
    );
    std::cout << "CPU Affinity: " << (cpu_ok ? "PASS" : "FAIL") << std::endl;
    if (!cpu_ok) {
        std::cout << "  Expected: " << expected_cpu_affinity << std::endl;
        std::cout << "  Actual:   " << actual_cpu_affinity << std::endl;
        all_passed = false;
    }

    // Check NUMA binding (simplified - check if any expected node matches)
    bool numa_ok = true;
    if (!expected_memory_binding.empty()) {
        if (actual_numa_nodes.empty()) {
            numa_ok = false;
        } else {
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
            numa_ok = found;
        }
    }
    std::cout << "NUMA Binding: " << (numa_ok ? "PASS" : "FAIL") << std::endl;
    if (!numa_ok) {
        std::cout << "  Expected: [";
        for (size_t i = 0; i < expected_memory_binding.size(); ++i) {
            if (i > 0)
                std::cout << ",";
            std::cout << expected_memory_binding[i];
        }
        std::cout << "]" << std::endl;
        std::cout << "  Actual:   [";
        for (size_t i = 0; i < actual_numa_nodes.size(); ++i) {
            if (i > 0)
                std::cout << ",";
            std::cout << actual_numa_nodes[i];
        }
        std::cout << "]" << std::endl;
        all_passed = false;
    }

    // Check UCX network devices
    std::string expected_ucx_devices;
    for (size_t i = 0; i < expected_network_devices.size(); ++i) {
        if (i > 0)
            expected_ucx_devices += ",";
        expected_ucx_devices += expected_network_devices[i];
    }
    bool ucx_ok = rapidsmpf::bootstrap::compare_device_lists(
        actual_ucx_net_devices, expected_ucx_devices
    );
    std::cout << "UCX_NET_DEVICES: " << (ucx_ok ? "PASS" : "FAIL") << std::endl;
    if (!ucx_ok) {
        std::cout << "  Expected: " << expected_ucx_devices << std::endl;
        std::cout << "  Actual:   " << actual_ucx_net_devices << std::endl;
        all_passed = false;
    }

    std::cout << "\n=== Result ===" << std::endl;
    if (all_passed) {
        std::cout << "All checks PASSED" << std::endl;
        return 0;
    } else {
        std::cout << "Some checks FAILED" << std::endl;
        return 1;
    }
}
