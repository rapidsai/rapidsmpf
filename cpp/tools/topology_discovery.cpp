/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <nvml.h>
#include <unistd.h>

namespace fs = std::filesystem;

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
        for (size_t i = 0; i < vec.size(); ++i) {
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
        for (size_t i = 0; i < vec.size(); ++i) {
            if (i > 0)
                oss << ", ";
            oss << "\"" << escape_string(vec[i]) << "\"";
        }
        oss << "]";
        return oss.str();
    }
};

/**
 * @brief GPU information.
 */
struct GpuInfo {
    unsigned int id;
    std::string name;
    std::string pci_bus_id;
    std::string uuid;
    int numa_node;
    std::string cpu_affinity_list;
    std::vector<int> cpu_cores;
    std::vector<int> memory_binding;
    std::vector<std::string> network_devices;
};

/**
 * @brief Network device information.
 */
struct NetworkDevice {
    std::string name;
    int numa_node;
    std::string pci_bus_id;
};

/**
 * @brief Read a file and return its content.
 *
 * @param path File to read.
 * @return The file content.
 */
std::string read_file_content(std::string const& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return "";
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();
    // Trim trailing newline
    if (!content.empty() && content.back() == '\n') {
        content.pop_back();
    }
    return content;
}

/**
 * @brief Parse CPU list string into a vector of core IDs.
 *
 * @param cpulist CPU list string (e.g., "0-31,128-159").
 * @return Vector of CPU core IDs.
 */
std::vector<int> parse_cpu_list(std::string const& cpulist) {
    std::vector<int> cores;
    if (cpulist.empty()) {
        return cores;
    }

    std::istringstream iss(cpulist);
    std::string token;

    while (std::getline(iss, token, ',')) {
        size_t dash_pos = token.find('-');
        if (dash_pos != std::string::npos) {
            // Range, e.g., "0-31"
            int start = std::stoi(token.substr(0, dash_pos));
            int end = std::stoi(token.substr(dash_pos + 1));
            for (int i = start; i <= end; ++i) {
                cores.push_back(i);
            }
        } else {
            // Single core, e.g., "5"
            cores.push_back(std::stoi(token));
        }
    }

    return cores;
}

/**
 * @brief Normalize PCI bus ID to standard format.
 *
 * @param pci_bus_id PCI bus ID to normalize.
 * @return Normalized PCI bus ID in format (0000:06:00.0).
 */
std::string normalize_pci_bus_id(std::string const& pci_bus_id) {
    // NVML may return format like "00000000:0A:00.0" but /sys uses "0000:0a:00.0"
    // Find the first colon and take the last 4 hex digits before it as domain
    size_t colon_pos = pci_bus_id.find(':');
    if (colon_pos == std::string::npos) {
        return pci_bus_id;
    }
    std::string domain = pci_bus_id.substr(0, colon_pos);
    if (domain.length() > 4) {
        domain = domain.substr(domain.length() - 4);
    }

    // Convert to lowercase
    std::string normalized_id = domain + pci_bus_id.substr(colon_pos);
    std::transform(
        normalized_id.begin(), normalized_id.end(), normalized_id.begin(), ::tolower
    );

    return normalized_id;
}

/**
 * @brief Get NUMA node from /sys.
 *
 * @param pci_bus_id PCI bus ID of the device.
 * @return NUMA node number, or -1 if not found.
 */
int get_numa_node_from_sys(std::string const& pci_bus_id) {
    std::string normalized_id = normalize_pci_bus_id(pci_bus_id);
    std::string path = "/sys/bus/pci/devices/" + normalized_id + "/numa_node";
    std::string content = read_file_content(path);
    if (content.empty()) {
        return -1;
    }
    try {
        return std::stoi(content);
    } catch (...) {
        return -1;
    }
}

/**
 * @brief Get CPU affinity list from /sys.
 *
 * @param pci_bus_id PCI bus ID of the device.
 * @return CPU affinity list string.
 */
std::string get_cpu_affinity_from_sys(std::string const& pci_bus_id) {
    std::string normalized_id = normalize_pci_bus_id(pci_bus_id);
    std::string path = "/sys/bus/pci/devices/" + normalized_id + "/local_cpulist";
    return read_file_content(path);
}

/**
 * @brief Get PCI bus ID from a device in /sys.
 *
 * @param device_path Path to the device in /sys.
 * @return PCI bus ID of the device.
 */
std::string get_pci_bus_id_from_device(std::string const& device_path) {
    fs::path device_link = fs::path(device_path) / "device";
    if (!fs::exists(device_link)) {
        return "";
    }

    try {
        fs::path real_path = fs::canonical(device_link);
        return real_path.filename().string();
    } catch (...) {
        return "";
    }
}

/**
 * @brief Discover network devices (InfiniBand/RoCE).
 *
 * @return Vector of discovered network devices.
 */
std::vector<NetworkDevice> discover_network_devices() {
    std::vector<NetworkDevice> devices;
    std::string ib_path = "/sys/class/infiniband";

    if (!fs::exists(ib_path)) {
        return devices;
    }

    try {
        for (auto const& entry : fs::directory_iterator(ib_path)) {
            if (!entry.is_directory()) {
                continue;
            }

            NetworkDevice dev;
            dev.name = entry.path().filename().string();

            // Get device's NUMA node and PCI bus ID
            std::string numa_path = entry.path().string() + "/device/numa_node";
            std::string numa_str = read_file_content(numa_path);
            dev.numa_node = numa_str.empty() ? -1 : std::stoi(numa_str);
            dev.pci_bus_id = get_pci_bus_id_from_device(entry.path().string());

            devices.push_back(dev);
        }
    } catch (std::exception const& e) {
        std::cerr << "Warning: Error discovering network devices: " << e.what()
                  << std::endl;
    }

    return devices;
}

/**
 * @brief Map network devices to GPUs based on NUMA proximity.
 *
 * @param gpu_numa_node NUMA node to query.
 * @param network_devices Network devices on the system.
 * @return Vector with device names closes to the NUMA node.
 */
std::vector<std::string> map_network_devices_to_gpu(
    int gpu_numa_node, std::vector<NetworkDevice> const& network_devices
) {
    std::vector<std::string> mapped_devices;

    // First, try to find devices on the same NUMA node
    for (auto const& dev : network_devices) {
        if (dev.numa_node == gpu_numa_node) {
            mapped_devices.push_back(dev.name);
        }
    }

    // If no devices found on the same NUMA node, return all devices
    // (this handles systems with shared network devices)
    if (mapped_devices.empty() && !network_devices.empty()) {
        for (auto const& dev : network_devices) {
            mapped_devices.push_back(dev.name);
        }
    }

    return mapped_devices;
}

/**
 * @brief Get system hostname.
 *
 * @return System hostname, empty string if hostname cannot be determined.
 */
std::string get_hostname() {
    char hostname[256];
    if (gethostname(hostname, sizeof(hostname)) == 0) {
        return std::string(hostname);
    }
    return "";
}

/**
 * @brief Count NUMA nodes on the system.
 *
 * @return Number of NUMA nodes.
 */
int count_numa_nodes() {
    std::string numa_path = "/sys/devices/system/node";
    int count = 0;

    if (!fs::exists(numa_path)) {
        return 0;
    }

    try {
        for (auto const& entry : fs::directory_iterator(numa_path)) {
            std::string name = entry.path().filename().string();
            if (name.rfind("node", 0) == 0) {  // starts with "node"
                count++;
            }
        }
    } catch (...) {
        return 0;
    }

    return count;
}

int main(int /* argc */, char** /* argv */) {
    nvmlReturn_t result = nvmlInit_v2();
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to initialize NVML: " << nvmlErrorString(result)
                  << std::endl;
        // Continue anyway to report system info even without GPUs
    }

    // Get GPU count
    unsigned int device_count = 0;
    bool nvml_available = false;
    if (result == NVML_SUCCESS) {
        result = nvmlDeviceGetCount_v2(&device_count);
        if (result != NVML_SUCCESS) {
            std::cerr << "Warning: Failed to get device count: "
                      << nvmlErrorString(result) << std::endl;
            device_count = 0;
        } else {
            nvml_available = true;
        }
    }

    // Get system information and discover network devices
    std::string hostname = get_hostname();
    int num_numa_nodes = count_numa_nodes();
    int num_network_devices = static_cast<int>(network_devices.size());
    std::vector<NetworkDevice> network_devices = discover_network_devices();

    // Collect GPU information
    std::vector<GpuInfo> gpus;

    for (unsigned int i = 0; i < device_count; ++i) {
        nvmlDevice_t device;
        result = nvmlDeviceGetHandleByIndex_v2(i, &device);
        if (result != NVML_SUCCESS) {
            std::cerr << "Warning: Failed to get handle for GPU " << i << ": "
                      << nvmlErrorString(result) << std::endl;
            continue;
        }

        GpuInfo gpu;
        gpu.id = i;

        // Get device name, PCI bus ID and UUID
        char name[NVML_DEVICE_NAME_BUFFER_SIZE];
        result = nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE);
        if (result == NVML_SUCCESS) {
            gpu.name = std::string(name);
        } else {
            gpu.name = "Unknown";
        }

        nvmlPciInfo_t pci_info;
        result = nvmlDeviceGetPciInfo_v3(device, &pci_info);
        if (result == NVML_SUCCESS) {
            gpu.pci_bus_id = std::string(pci_info.busId);
        } else {
            std::cerr << "Warning: Failed to get PCI info for GPU " << i << std::endl;
            continue;
        }

        char uuid[NVML_DEVICE_UUID_BUFFER_SIZE];
        result = nvmlDeviceGetUUID(device, uuid, NVML_DEVICE_UUID_BUFFER_SIZE);
        if (result == NVML_SUCCESS) {
            gpu.uuid = std::string(uuid);
        } else {
            gpu.uuid = "Unknown";
        }

        // Get NUMA node and CPU affinity from /sys
        gpu.numa_node = get_numa_node_from_sys(gpu.pci_bus_id);
        gpu.cpu_affinity_list = get_cpu_affinity_from_sys(gpu.pci_bus_id);
        gpu.cpu_cores = parse_cpu_list(gpu.cpu_affinity_list);

        // Memory binding is typically the same as NUMA node
        if (gpu.numa_node >= 0) {
            gpu.memory_binding.push_back(gpu.numa_node);
        }

        // Map network devices to this GPU
        gpu.network_devices = map_network_devices_to_gpu(gpu.numa_node, network_devices);

        gpus.push_back(gpu);
    }

    if (nvml_available) {
        nvmlShutdown();
    }

    // Generate JSON output
    std::cout << "{\n";
    std::cout << "  \"system\": {\n";
    std::cout << "    \"hostname\": \"" << JsonBuilder::escape_string(hostname)
              << "\",\n";
    std::cout << "    \"num_gpus\": " << device_count << ",\n";
    std::cout << "    \"num_numa_nodes\": " << num_numa_nodes << ",\n";
    std::cout << "    \"num_network_devices\": " << num_network_devices << "\n";
    std::cout << "  },\n";

    std::cout << "  \"gpus\": [\n";
    for (size_t i = 0; i < gpus.size(); ++i) {
        auto const& gpu = gpus[i];
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
        if (i < gpus.size() - 1) {
            std::cout << ",";
        }
        std::cout << "\n";
    }
    std::cout << "  ],\n";

    std::cout << "  \"network_devices\": [\n";
    for (size_t i = 0; i < network_devices.size(); ++i) {
        auto const& dev = network_devices[i];
        std::cout << "    {\n";
        std::cout << "      \"name\": \"" << JsonBuilder::escape_string(dev.name)
                  << "\",\n";
        std::cout << "      \"numa_node\": " << dev.numa_node << ",\n";
        std::cout << "      \"pci_bus_id\": \""
                  << JsonBuilder::escape_string(dev.pci_bus_id) << "\"\n";
        std::cout << "    }";
        if (i < network_devices.size() - 1) {
            std::cout << ",";
        }
        std::cout << "\n";
    }
    std::cout << "  ]\n";
    std::cout << "}\n";

    return 0;
}
