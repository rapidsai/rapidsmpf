/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include <sstream>
#include <string>

#include <cucascade/memory/topology_discovery.hpp>

/**
 * @brief Simple JSON builder to avoid external dependencies.
 */
class JsonBuilder {
  public:
    static std::string escape_string(std::string const& str) {
        std::string result;
        for (char c : str) {
            switch (c) {
            case '"':
                result += "\\\"";
                break;
            case '\\':
                result += "\\\\";
                break;
            case '\b':
                result += "\\b";
                break;
            case '\f':
                result += "\\f";
                break;
            case '\n':
                result += "\\n";
                break;
            case '\r':
                result += "\\r";
                break;
            case '\t':
                result += "\\t";
                break;
            default:
                result += c;
                break;
            }
        }
        return result;
    }

    static std::string to_json_array(std::vector<int> const& vec) {
        std::ostringstream oss;
        oss << "[";
        for (std::size_t i = 0; i < vec.size(); ++i) {
            if (i > 0)
                oss << ", ";
            oss << vec[i];
        }
        oss << "]";
        return oss.str();
    }

    static std::string to_json_array(std::vector<std::string> const& vec) {
        std::ostringstream oss;
        oss << "[";
        for (std::size_t i = 0; i < vec.size(); ++i) {
            if (i > 0)
                oss << ", ";
            oss << "\"" << escape_string(vec[i]) << "\"";
        }
        oss << "]";
        return oss.str();
    }
};

/**
 * @brief Output topology information as JSON
 */
void output_json(cucascade::memory::system_topology_info const& topology) {
    std::cout << "{\n";
    std::cout << "  \"system\": {\n";
    std::cout << "    \"hostname\": \"" << JsonBuilder::escape_string(topology.hostname)
              << "\",\n";
    std::cout << "    \"num_gpus\": " << topology.num_gpus << ",\n";
    std::cout << "    \"num_numa_nodes\": " << topology.num_numa_nodes << ",\n";
    std::cout << "    \"num_network_devices\": " << topology.num_network_devices << "\n";
    std::cout << "  },\n";

    std::cout << "  \"gpus\": [\n";
    for (std::size_t i = 0; i < topology.gpus.size(); ++i) {
        auto const& gpu = topology.gpus[i];
        std::cout << "    {\n";
        std::cout << "      \"id\": " << gpu.id << ",\n";
        std::cout << "      \"name\": \"" << JsonBuilder::escape_string(gpu.name)
                  << "\",\n";
        std::cout << "      \"pci_bus_id\": \""
                  << JsonBuilder::escape_string(gpu.pci_bus_id) << "\",\n";
        std::cout << "      \"uuid\": \"" << JsonBuilder::escape_string(gpu.uuid)
                  << "\",\n";
        std::cout << "      \"numa_node\": " << gpu.numa_node << ",\n";
        std::cout << "      \"cpu_affinity\": {\n";
        std::cout << "        \"cpulist\": \""
                  << JsonBuilder::escape_string(gpu.cpu_affinity_list) << "\",\n";
        std::cout << "        \"cores\": " << JsonBuilder::to_json_array(gpu.cpu_cores)
                  << "\n";
        std::cout << "      },\n";
        std::cout << "      \"memory_binding\": "
                  << JsonBuilder::to_json_array(gpu.memory_binding) << ",\n";
        std::cout << "      \"network_devices\": "
                  << JsonBuilder::to_json_array(gpu.network_devices) << "\n";
        std::cout << "    }";
        if (i < topology.gpus.size() - 1) {
            std::cout << ",";
        }
        std::cout << "\n";
    }
    std::cout << "  ],\n";

    std::cout << "  \"network_devices\": [\n";
    for (std::size_t i = 0; i < topology.network_devices.size(); ++i) {
        auto const& dev = topology.network_devices[i];
        std::cout << "    {\n";
        std::cout << "      \"name\": \"" << JsonBuilder::escape_string(dev.name)
                  << "\",\n";
        std::cout << "      \"numa_node\": " << dev.numa_node << ",\n";
        std::cout << "      \"pci_bus_id\": \""
                  << JsonBuilder::escape_string(dev.pci_bus_id) << "\"\n";
        std::cout << "    }";
        if (i < topology.network_devices.size() - 1) {
            std::cout << ",";
        }
        std::cout << "\n";
    }
    std::cout << "  ]\n";
    std::cout << "}\n";
}

int main(int /* argc */, char** /* argv */) {
    cucascade::memory::topology_discovery discovery;

    if (!discovery.discover()) {
        std::cerr << "Failed to discover system topology" << std::endl;
        return 1;
    }

    output_json(discovery.get_topology());

    return 0;
}
