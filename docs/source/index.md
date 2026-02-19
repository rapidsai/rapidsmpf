# RapidsMPF documentation

Building high-performance GPU pipelines is hard. Each stage must move data efficiently between GPUs and processes, synchronize work, and manage limited device memory.
RapidsMPF, including its nascent Streaming Engine and Out-of-Core (OOC) shuffle, provides a unified framework for asynchronous, multi-GPU pipelines using simple streaming primitives — {term}`Channel`s, {term}`Actor`s, and {term}`Message`s — built on RAPIDS components: [rmm](https://docs.rapids.ai/api/rmm/nightly), [libcudf](https://docs.rapids.ai/api/libcudf/nightly/), and [ucxx](https://docs.rapids.ai/api/ucxx/nightly/).

RapidsMPF's design leverages Explicit Data Parallelism (SPMD-style coordination) combined with a local CSP-style streaming model, enabling the engine to overlap I/O, computation, and communication. This makes it possible to handle out-of-core processing efficiently (via {term}`Spilling`) and integrate seamlessly with frontend query engines such as Polars.
The result is clean, composable, and scalable GPU streaming — from single-node prototypes to large-scale, multi-GPU deployments. See the {doc}`glossary` for definitions of key concepts.

```{toctree}
---
maxdepth: 2
caption: Contents:
---
getting-started.md
background/index.md
python/index.md
cpp/index.md
configuration.md
glossary.md
```
