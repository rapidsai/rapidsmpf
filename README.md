# RapidsMPF

Collection of multi-GPU, distributed memory algorithms. RapidsMPF provides a unified
framework for asynchronous, multi-GPU pipelines using simple streaming primitives built
using NVIDIA CUDA-X components.

## Documentation

- [Getting Started](https://docs.nvidia.com/rapidsmpf/latest/getting-started/)
- [Background](https://docs.nvidia.com/rapidsmpf/latest/background/)
- [Configuration Options](https://docs.nvidia.com/rapidsmpf/latest/configuration/)
- [Python API Reference](https://docs.nvidia.com/rapidsmpf/latest/python/api/)
- [C++ API Reference](https://docs.rapids.ai/api/librapidsmpf/nightly/)
- [Glossary](https://docs.nvidia.com/rapidsmpf/latest/glossary/)

## Build from Source

```bash
git clone https://github.com/rapidsai/rapidsmpf.git
cd rapidsmpf
mamba env create --name rapidsmpf-dev --file conda/environments/all_cuda-133_arch-$(uname -m).yaml
./build.sh
```

See the [Getting Started guide](https://docs.nvidia.com/rapidsmpf/latest/getting-started/)
for debug builds, AddressSanitizer, MPI/UCX test suites, and rrun launcher details.
