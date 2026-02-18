/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <cucascade/memory/topology_discovery.hpp>

#include <rapidsmpf/bootstrap/utils.hpp>
#include <rapidsmpf/system_info.hpp>

namespace {

/**
 * @brief Command line arguments.
 */
struct Arguments {
    std::string json_file;  ///< Path to JSON topology file (empty if not provided).
    int gpu_id = -1;  ///< GPU ID to validate (-1 if not provided).
    bool validate = false;  ///< Whether to perform validation.
    bool show_help = false;  ///< Whether to show help message.
};

/**
 * @brief Actual binding configuration collected from the system.
 */
struct ActualBinding {
    int rank = -1;  ///< Process rank (-1 if not available).
    int gpu_id = -1;  ///< GPU device ID (-1 if not available).
    std::string gpu_pci_bus_id;  ///< GPU PCI bus ID.
    std::string cpu_affinity;  ///< CPU affinity string.
    std::vector<int> numa_nodes;  ///< NUMA node IDs.
    std::string ucx_net_devices;  ///< UCX_NET_DEVICES environment variable.
};

/**
 * @brief Expected binding configuration for validation.
 */
struct ExpectedBinding {
    std::string cpu_affinity;  ///< Expected CPU affinity.
    std::vector<int> memory_binding;  ///< Expected NUMA nodes for memory binding.
    std::vector<std::string> network_devices;  ///< Expected network devices.
};

/**
 * @brief Results of validation checks.
 */
struct ValidationResult {
    bool cpu_ok = true;  ///< CPU affinity check passed.
    bool numa_ok = true;  ///< NUMA binding check passed.
    bool ucx_ok = true;  ///< UCX network devices check passed.
    std::string
        expected_ucx_devices;  ///< Expected UCX devices as comma-separated string.

    /**
     * @brief Check if all validations passed.
     */
    [[nodiscard]] bool all_passed() const {
        return cpu_ok && numa_ok && ucx_ok;
    }
};

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

/**
 * @brief Get PCI bus ID for the current CUDA device.
 *
 * When CUDA_VISIBLE_DEVICES is set (as done by rrun), CUDA only sees
 * the assigned GPU as logical device 0, regardless of the physical GPU ID.
 * This function queries the PCI bus ID of the current CUDA device.
 *
 * @return PCI bus ID string, or "(unknown)" on error.
 */
std::string get_gpu_pci_bus_id() {
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        return "(unknown)";
    }
    std::string pci_bus_id(16, '\0');  // Preallocate space for the PCI bus ID
    err = cudaDeviceGetPCIBusId(pci_bus_id.data(), pci_bus_id.size(), device);
    if (err != cudaSuccess) {
        return "(unknown)";
    }
    // Trim to actual string length
    return pci_bus_id.substr(0, pci_bus_id.find('\0'));
}

/**
 * @brief Parse command line arguments.
 *
 * @param argc Argument count.
 * @param argv Argument values.
 * @return Parsed arguments.
 */
Arguments parse_arguments(int argc, char* argv[]) {
    Arguments args;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--json" && i + 1 < argc) {
            args.json_file = argv[++i];
            args.validate = true;
        } else if (arg == "--gpu-id" && i + 1 < argc) {
            args.gpu_id = std::stoi(argv[++i]);
            args.validate = true;
        } else if (arg == "--help" || arg == "-h") {
            args.show_help = true;
        }
    }

    return args;
}

/**
 * @brief Print help message.
 *
 * @param program_name Name of the program (argv[0]).
 */
void print_help(char const* program_name) {
    std::cout << "Usage: " << program_name
              << " [--json <topology.json>] [--gpu-id <id>]\n"
              << "\n"
              << "Options:\n"
              << "  --json <file>    Read expected values from JSON file\n"
              << "  --gpu-id <id>    Use TopologyDiscovery API for GPU ID\n"
              << "  --help, -h       Show this help message\n"
              << "\n"
              << "If neither --json nor --gpu-id is provided, "
              << "just reports current configuration.\n"
              << "\n"
              << "Examples usage to validate resource binding on GPU 3 only:\n"
              << "  $ topology_discovery > topology.json\n"
              << "  $ rrun -n 1 -g 3 check_resource_binding --json topology.json\n"
              << "    === Topology Binding Test ===\n"
              << "    Rank: 0\n"
              << "    GPU ID: 3\n"
              << "    GPU PCI Bus ID: 00000000:41:00.0\n"
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
}

/**
 * @brief Collect actual binding configuration from the system.
 *
 * This function performs potentially slow operations like querying CPU affinity,
 * NUMA nodes, and GPU PCI bus ID (which may trigger CUDA initialization).
 *
 * @param gpu_id_hint GPU ID hint from command line (-1 to auto-detect).
 * @return Collected binding information.
 */
ActualBinding collect_actual_binding(int gpu_id_hint) {
    ActualBinding binding;

    binding.cpu_affinity = rapidsmpf::bootstrap::get_current_cpu_affinity();
    binding.numa_nodes = rapidsmpf::get_current_numa_nodes();
    binding.ucx_net_devices = rapidsmpf::bootstrap::get_ucx_net_devices();
    binding.rank = rapidsmpf::bootstrap::get_rank();

    binding.gpu_id =
        (gpu_id_hint >= 0) ? gpu_id_hint : rapidsmpf::bootstrap::get_gpu_id();

    if (binding.gpu_id >= 0) {
        binding.gpu_pci_bus_id = get_gpu_pci_bus_id();
    }

    return binding;
}

/**
 * @brief Collect expected binding configuration for validation.
 *
 * Reads expected values from either a JSON file or uses TopologyDiscovery API.
 *
 * @param json_file Path to JSON topology file (empty to use TopologyDiscovery).
 * @param gpu_id GPU ID to look up in the topology.
 * @return Expected binding if successful, std::nullopt on error.
 */
std::optional<ExpectedBinding> collect_expected_binding(
    std::string const& json_file, int gpu_id
) {
    ExpectedBinding expected;

    if (!json_file.empty()) {
        if (!extract_gpu_info_from_json(
                json_file,
                gpu_id,
                expected.cpu_affinity,
                expected.memory_binding,
                expected.network_devices
            ))
        {
            return std::nullopt;
        }
    } else if (gpu_id >= 0) {
        cucascade::memory::topology_discovery discovery;
        if (!discovery.discover()) {
            std::cerr << "Error: Failed to discover topology" << std::endl;
            return std::nullopt;
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
            return std::nullopt;
        }

        expected.cpu_affinity = it->cpu_affinity_list;
        expected.memory_binding = it->memory_binding;
        expected.network_devices = it->network_devices;
    } else {
        std::cerr << "Error: Must provide --json or --gpu-id for validation" << std::endl;
        return std::nullopt;
    }

    return expected;
}

/**
 * @brief Validate actual binding against expected values.
 *
 * @param actual Actual binding configuration.
 * @param expected Expected binding configuration.
 * @return Validation results.
 */
ValidationResult validate_binding(
    ActualBinding const& actual, ExpectedBinding const& expected
) {
    ValidationResult result;

    // Check CPU affinity
    result.cpu_ok = rapidsmpf::bootstrap::compare_cpu_affinity(
        actual.cpu_affinity, expected.cpu_affinity
    );

    // Check NUMA binding (check if any actual node is in expected list)
    if (!expected.memory_binding.empty()) {
        if (actual.numa_nodes.empty()) {
            result.numa_ok = false;
        } else {
            bool found = false;
            for (int actual_node : actual.numa_nodes) {
                if (std::find(
                        expected.memory_binding.begin(),
                        expected.memory_binding.end(),
                        actual_node
                    )
                    != expected.memory_binding.end())
                {
                    found = true;
                    break;
                }
            }
            result.numa_ok = found;
        }
    }

    // Build expected UCX devices string and check
    for (size_t i = 0; i < expected.network_devices.size(); ++i) {
        if (i > 0) {
            result.expected_ucx_devices += ",";
        }
        result.expected_ucx_devices += expected.network_devices[i];
    }
    result.ucx_ok = rapidsmpf::bootstrap::compare_device_lists(
        actual.ucx_net_devices, result.expected_ucx_devices
    );

    return result;
}

/**
 * @brief Format output string for the binding test.
 *
 * Builds the complete output string in memory to minimize interleaved output
 * when multiple processes print simultaneously.
 *
 * @param actual Actual binding configuration.
 * @param expected Expected binding configuration (nullptr if not validating).
 * @param validation Validation results (nullptr if not validating).
 * @return Formatted output string.
 */
std::string format_output(
    ActualBinding const& actual,
    ExpectedBinding const* expected,
    ValidationResult const* validation
) {
    std::ostringstream output;

    // Output current configuration
    output << "=== Topology Binding Test ===" << std::endl;
    if (actual.rank >= 0) {
        output << "Rank: " << actual.rank << std::endl;
    }
    if (actual.gpu_id >= 0) {
        output << "GPU ID: " << actual.gpu_id << std::endl;
        output << "GPU PCI Bus ID: " << actual.gpu_pci_bus_id << std::endl;
    }
    output << "CPU Affinity: "
           << (actual.cpu_affinity.empty() ? "(none)" : actual.cpu_affinity) << std::endl;
    output << "NUMA Nodes: ";
    if (actual.numa_nodes.empty()) {
        output << "(none)";
    } else {
        for (size_t i = 0; i < actual.numa_nodes.size(); ++i) {
            if (i > 0) {
                output << ",";
            }
            output << actual.numa_nodes[i];
        }
    }
    output << std::endl;
    output << "UCX_NET_DEVICES: "
           << (actual.ucx_net_devices.empty() ? "(not set)" : actual.ucx_net_devices)
           << std::endl;

    // Output validation results if available
    if (expected != nullptr && validation != nullptr) {
        output << "\n=== Validation ===" << std::endl;

        output << "CPU Affinity: " << (validation->cpu_ok ? "PASS" : "FAIL") << std::endl;
        if (!validation->cpu_ok) {
            output << "  Expected: " << expected->cpu_affinity << std::endl;
            output << "  Actual:   " << actual.cpu_affinity << std::endl;
        }

        output << "NUMA Binding: " << (validation->numa_ok ? "PASS" : "FAIL")
               << std::endl;
        if (!validation->numa_ok) {
            output << "  Expected: [";
            for (size_t i = 0; i < expected->memory_binding.size(); ++i) {
                if (i > 0) {
                    output << ",";
                }
                output << expected->memory_binding[i];
            }
            output << "]" << std::endl;
            output << "  Actual:   [";
            for (size_t i = 0; i < actual.numa_nodes.size(); ++i) {
                if (i > 0) {
                    output << ",";
                }
                output << actual.numa_nodes[i];
            }
            output << "]" << std::endl;
        }

        output << "UCX_NET_DEVICES: " << (validation->ucx_ok ? "PASS" : "FAIL")
               << std::endl;
        if (!validation->ucx_ok) {
            output << "  Expected: " << validation->expected_ucx_devices << std::endl;
            output << "  Actual:   " << actual.ucx_net_devices << std::endl;
        }

        output << "\n=== Result ===" << std::endl;
        if (validation->all_passed()) {
            output << "All checks PASSED" << std::endl;
        } else {
            output << "Some checks FAILED" << std::endl;
        }
    }

    return output.str();
}

}  // namespace

/**
 * @brief Print or validate binding configuration.
 *
 * See top of this file for more details.
 */
int main(int argc, char* argv[]) {
    // Parse command line arguments
    Arguments args = parse_arguments(argc, argv);

    if (args.show_help) {
        print_help(argv[0]);
        return 0;
    }

    ActualBinding actual = collect_actual_binding(args.gpu_id);

    std::optional<ExpectedBinding> expected;
    std::optional<ValidationResult> validation;

    if (args.validate) {
        expected = collect_expected_binding(args.json_file, actual.gpu_id);
        if (!expected) {
            return 1;
        }
        validation = validate_binding(actual, *expected);
    }

    std::string output = format_output(
        actual, expected ? &(*expected) : nullptr, validation ? &(*validation) : nullptr
    );
    std::cout << output << std::flush;

    return (validation && !validation->all_passed()) ? 1 : 0;
}
