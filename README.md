# RapidsMPF

Collection of multi-GPU, distributed memory algorithms. RapidsMPF provides a unified
framework for asynchronous, multi-GPU pipelines using simple streaming primitives built
on RAPIDS components.

## Documentation

- [Getting Started](https://docs.rapids.ai/api/rapidsmpf/stable/getting-started/) ([nightly](https://docs.rapids.ai/api/rapidsmpf/nightly/getting-started/))
- [Background](https://docs.rapids.ai/api/rapidsmpf/stable/background/) ([nightly](https://docs.rapids.ai/api/rapidsmpf/nightly/background/))
- [Configuration Options](https://docs.rapids.ai/api/rapidsmpf/stable/configuration/) ([nightly](https://docs.rapids.ai/api/rapidsmpf/nightly/configuration/))
- [Python API Reference](https://docs.rapids.ai/api/rapidsmpf/stable/python/api/) ([nightly](https://docs.rapids.ai/api/rapidsmpf/nightly/python/api/))
- [C++ API Reference](https://docs.rapids.ai/api/librapidsmpf/stable/) ([nightly](https://docs.rapids.ai/api/librapidsmpf/nightly/))
- [Glossary](https://docs.rapids.ai/api/rapidsmpf/stable/glossary/) ([nightly](https://docs.rapids.ai/api/rapidsmpf/nightly/glossary/))

## Build from Source

```bash
git clone https://github.com/rapidsai/rapidsmpf.git
cd rapidsmpf
mamba env create --name rapidsmpf-dev --file conda/environments/all_cuda-131_arch-$(uname -m).yaml
./build.sh
```

See the [Getting Started guide](https://docs.rapids.ai/api/rapidsmpf/stable/getting-started/) ([nightly](https://docs.rapids.ai/api/rapidsmpf/nightly/getting-started/))
for debug builds, AddressSanitizer, MPI/UCX test suites, and rrun launcher details.
