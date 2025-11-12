# RapidsMPF Tools

This directory contains utility tools for the RapidsMPF project.

## Topology Discovery Tool

### Overview

The `topology_discovery` tool programmatically discovers GPU, CPU, NUMA, and network device topology information on a system. It replaces hardcoded assumptions in binding scripts like those used with OpenMPI/`mpirun` with dynamic, system-aware discovery.

### Features

- **GPU Discovery**: Uses NVML to enumerate all GPUs and retrieve device properties
- **NUMA Topology**: Queries `/sys` filesystem to determine NUMA node associations
- **CPU Affinity**: Discovers which CPU cores are local to each GPU
- **Network Devices**: Maps InfiniBand/RoCE network devices to GPUs based on NUMA proximity
- **JSON Output**: Produces easy-to-parse JSON for integration with scripts

### Usage

Run the tool to output system topology as JSON:

```bash
./cpp/build/tools/topology_discovery > system_topology.json
```

### Output Format

The tool outputs JSON with the following structure:

```json
{
  "system": {
    "hostname": "dgx13",
    "num_gpus": 8,
    "num_numa_nodes": 2,
    "num_network_devices": 4
  },
  "gpus": [
    {
      "id": 0,
      "name": "Tesla V100-SXM2-32GB",
      "pci_bus_id": "00000000:06:00.0",
      "uuid": "GPU-9baca7f5-0f2f-01ac-6b05-8da14d6e9005",
      "numa_node": 0,
      "cpu_affinity": {
        "cpulist": "0-19,40-59",
        "cores": [0, 1, 2, ..., 59]
      },
      "memory_binding": [0],
      "network_devices": ["mlx5_1", "mlx5_0"]
    }
  ],
  "network_devices": [
    {
      "name": "mlx5_0",
      "numa_node": 0,
      "pci_bus_id": "0000:05:00.0"
    }
  ]
}
```
