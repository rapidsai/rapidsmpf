# AGENTS.md - RapidsMPF Development Guide

RapidsMPF (RAPIDS Multi-Process / Multi-GPU Framework) is a collection of
multi-GPU, distributed-memory algorithms and streaming primitives built on
RAPIDS components (RMM, libcudf, libcudacxx) and pluggable communicators
(MPI, UCXX, single-process). The library is consumed from C++ and from
Python (Cython bindings).

## Safety Rules for Agents

- **Minimal diffs**: Change only what's necessary; avoid drive-by refactors.
- **No mass reformatting**: Don't run formatters over unrelated code.
- **No API invention**: Align with existing RapidsMPF patterns and documented APIs.
- **Don't bypass CI**: Don't suggest skipping checks or using `--no-verify`.
- **CUDA/GPU hygiene**: Keep operations stream-ordered, use RMM allocators,
  and never call `cudaMemcpyAsync` directly — use
  `rapidsmpf::cuda_memcpy_async` from
  `cpp/include/rapidsmpf/memory/cuda_memcpy_async.hpp` (enforced by the
  `use-rapidsmpf-cuda-memcpy-async` pre-commit hook).
- **Wrap every CUDA call**: Use `RAPIDSMPF_CUDA_TRY` (or
  `RAPIDSMPF_CUDA_TRY_FATAL` in destructors / `noexcept` paths, or
  `RAPIDSMPF_CUDA_TRY_ALLOC` for allocations).
- **Communicator safety**: Validate ranks, match collectives on every rank
  with identical arguments, and do not perform blocking work on the
  `ProgressThread`.

### Before Finalizing a Change

Ask yourself:

- What scenarios must be covered? (happy path, edge cases, failure modes,
  single-rank and multi-rank, spill-under-pressure, slow/disconnected peers)
- What's the expected behavior contract? (inputs/outputs, errors, stream
  semantics)
- Where should tests live? (C++ gtests under `cpp/tests/`, Python pytests
  under `python/rapidsmpf/rapidsmpf/tests/`)

## Code Review Guidelines

For AI-assisted code review (CodeRabbit), see language-specific review
guidelines:

- C++/CUDA: [cpp/REVIEW_GUIDELINES.md](cpp/REVIEW_GUIDELINES.md)
- Python (incl. Cython): [python/REVIEW_GUIDELINES.md](python/REVIEW_GUIDELINES.md)

## Build Commands

### Standard environment (requires conda env from `conda/environments/`)

```bash
./build.sh                      # Build and install librapidsmpf then rapidsmpf
./build.sh librapidsmpf         # Build and install C++ library only
./build.sh rapidsmpf            # Build and install Python package only
./build.sh -g                   # Debug build
./build.sh -n librapidsmpf      # Build without install
./build.sh --pydevelop          # Install Python in editable mode
./build.sh --asan               # Enable AddressSanitizer (C++ and Python)
./build.sh --no-clang-tidy      # Disable clang-tidy build checks
./build.sh clean                # Remove build artifacts
```

Pass additional CMake flags with `--cmake-args="..."` (escape quotes).

### Devcontainer

The `.devcontainer/` setup provides `build-all -j0` which builds the full
stack with sccache. See [`.devcontainer/`](.devcontainer/) for details.

## Test Commands

### C++ tests (GoogleTest)

C++ tests are bundled into a small number of executables under
`cpp/build/gtests/`. Each test suite is registered against the
executables for which it makes sense:

| Executable | Built when | How to run | Coverage |
|-----------|-----------|-----------|---------|
| `single_tests` | always | in-process (no `mpirun`) | suites that only need the single-rank (in-process) communicator |
| `mpi_tests` | `RAPIDSMPF_HAVE_MPI` | `mpirun -np N gtests/mpi_tests` | suites that work with the MPI communicator (single rank and multi-rank, including `N == 1`) |
| `ucxx_tests` | `RAPIDSMPF_HAVE_MPI` + `RAPIDSMPF_HAVE_UCXX` | `mpirun -np N gtests/ucxx_tests` (UCXX bootstraps over MPI) | suites that work with the UCXX communicator (single rank and multi-rank) |
| `rrun_tests` | always | `rrun -n N gtests/rrun_tests` (NOT via ctest / mpirun) | `rrun` launcher, topology discovery, resource binding |
| `bootstrap_tests` | always | in-process | bootstrap backends (socket, file, optional SLURM) |

Run them via `ctest` (which knows the per-rank invocations) or directly:

```bash
# All registered cases. ctest registers mpi_tests / ucxx_tests at np
# = 1, 2, 3, 4, 5, 8 (one case per np), so e.g. `mpi_tests_4` runs
# `mpirun -np 4 gtests/mpi_tests`.
ctest --test-dir cpp/build/tests --output-on-failure

# Filter by regex (e.g. only the MPI multi-rank cases)
ctest --test-dir cpp/build/tests -R 'mpi_tests_'

# Run a single executable directly (single-rank / in-process)
./cpp/build/gtests/single_tests
./cpp/build/gtests/single_tests --gtest_filter='Shuffler.*'

# Run an MPI/UCXX executable directly at chosen rank count
mpirun --map-by node --bind-to none -np 4 ./cpp/build/gtests/mpi_tests
mpirun --map-by node --bind-to none -np 2 ./cpp/build/gtests/ucxx_tests

# Run rrun_tests via rrun (not mpirun, not ctest)
rrun -n 1 ./cpp/build/gtests/rrun_tests
```

When adding a new test suite, register it against every executable
whose communicator can support it. Suites that exercise the
in-process / single-rank communicator only belong in `single_tests`;
suites that need multi-rank semantics belong in `mpi_tests` (and
`ucxx_tests` if the path exercises UCXX). Suites that need both
single and multi-rank end up in all three.

### C++ compute-sanitizer (memcheck)

See [`ci/test_cpp_memcheck.sh`](ci/test_cpp_memcheck.sh) for the exact
invocation used in CI; the suppression list is at
[`cpp/compute-sanitizer-suppressions.xml`](cpp/compute-sanitizer-suppressions.xml).

### Python tests (pytest)

Most Python tests are written against the parameterized `comm` fixture
in [`python/rapidsmpf/rapidsmpf/tests/conftest.py`](python/rapidsmpf/rapidsmpf/tests/conftest.py)
— that fixture sets up a per-test communicator over both `mpi`
and `ucxx` backends. UCXX is bootstrapped over MPI, so both paths
require launching pytest under `mpirun`.

Single-rank tests can be run as plain `pytest`; anything that uses the
`comm` fixture (or otherwise needs MPI / UCXX) must be spawned with
`mpirun`, mirroring the C++ `mpi_tests` / `ucxx_tests` pattern. CI
runs the suite at np = 1, 2, 3, 4, 5, 8 — see
[`ci/run_pytests.sh`](ci/run_pytests.sh).

```bash
# Run from python/rapidsmpf/rapidsmpf/ so the conftest is discovered
cd python/rapidsmpf/rapidsmpf

# Single-rank pytest (skips MPI-only tests when not under mpirun)
pytest tests/ -v

# Multi-rank pytest (MPI + UCXX): launch pytest under mpirun
mpirun --map-by node --bind-to none -np 4 python -m pytest tests -v

# Single test file under mpirun
mpirun --map-by node --bind-to none -np 4 \
  python -m pytest tests/streaming/test_shuffler.py -v

# Skip MPI-tagged tests (the conftest exposes --disable-mpi); used in
# CI for the UCXX-polling-mode pass
mpirun --map-by node --bind-to none -np 4 \
  python -m pytest --disable-mpi tests -v
```

OpenMPI environment knobs used by CI (set them locally if you're
running as root or hit CUDA-IPC issues):

```bash
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export OMPI_MCA_opal_cuda_support=1
```

The `rrun` launcher and its Python binding under
[`python/rapidsmpf/rapidsmpf/rrun/`](python/rapidsmpf/rapidsmpf/rrun/)
exist for production launches (topology discovery, resource binding)
and are exercised by `rrun_tests`; the Python test suite itself is
spawned with `mpirun`, not `rrun`.

## Lint and Format

Always use `pre-commit` to run linters and formatters:

```bash
pre-commit run --all-files                  # Run all hooks (recommended)
pre-commit run clang-format --all-files     # C++ formatting only
pre-commit run ruff-check --all-files       # Python linting only
pre-commit run ruff-format --all-files      # Python formatting only
pre-commit run cython-lint --all-files      # Cython linting only
pre-commit run mypy --all-files             # Python type checking
pre-commit run shellcheck --all-files       # Shell scripts
```

`clang-tidy` runs as part of the C++ build (disable with
`./build.sh --no-clang-tidy`).

## Reproducing CI Locally

See [`.agents/skills/reproduce-ci-locally/SKILL.md`](.agents/skills/reproduce-ci-locally/SKILL.md)
for the full procedure (Docker images, `RAPIDS_*` env vars, GH token,
build/test scripts under `ci/`).

## Code Style Guidelines

`pre-commit run --all-files` is the source of truth for formatting and
style.

### Naming Conventions

- **C++ classes**: `PascalCase` for types in the public API (e.g.,
  `RmmResourceAdaptor`, `BufferResource`, `Shuffler`), with `snake_case`
  members and free functions.
- **C++ functions / methods / variables**: `snake_case`.
- **C++ constants / macros**: `SCREAMING_SNAKE_CASE`, with the
  `RAPIDSMPF_` prefix for project macros.
- **C++ namespaces**: `rapidsmpf::*` (with `rapidsmpf::detail::*` for
  implementation details).
- **Python**: `snake_case` for functions and variables; `PascalCase` for
  classes.

### Error Handling

- **C++**: Use the macros from
  [`cpp/include/rapidsmpf/error.hpp`](cpp/include/rapidsmpf/error.hpp):
  - `RAPIDSMPF_EXPECTS(cond, msg[, exception_type])` for precondition checks
    (the condition must be a pure predicate — no side effects)
  - `RAPIDSMPF_FAIL(msg[, exception_type])` for unreachable / failure paths
  - `RAPIDSMPF_CUDA_TRY(cuda_call[, exception_type])` for every CUDA call
  - `RAPIDSMPF_CUDA_TRY_ALLOC(cuda_call[, num_bytes])` for CUDA allocations
  - `RAPIDSMPF_CUDA_TRY_FATAL(cuda_call)` /
    `RAPIDSMPF_EXPECTS_FATAL(cond, msg)` in destructors / `noexcept` paths
  - Exception types: `rapidsmpf::cuda_error`, `rapidsmpf::bad_alloc`
- **Python**: Raise standard Python exceptions or
  `rapidsmpf`-defined exceptions; C++ exceptions cross the Cython boundary
  via `except +`.

### File Headers (SPDX, required)

C++ and CUDA:

```cpp
/*
 * SPDX-FileCopyrightText: Copyright (c) YEAR, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
```

Python / Cython:

```python
# SPDX-FileCopyrightText: Copyright (c) YEAR, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
```

The `verify-copyright` pre-commit hook maintains the year range
(`YEAR_CREATED-YEAR_LAST_MODIFIED`) automatically, so just use the
current year for new files and let the hook update existing ones.

### Documentation

- **C++**: Doxygen comments (`/** ... */`) for all public APIs, with
  `@brief`, `@param`, `@return`, `@throw`, `@tparam` as applicable.
  Enforced by the `doxygen-check` pre-commit hook on `cpp/include/`.
- **Python**: NumPy-style docstrings on all public functions and classes.

## Project Structure

```text
cpp/        # C++ / CUDA source, public headers, GoogleTest suites, benchmarks
python/     # librapidsmpf wheel + rapidsmpf Cython bindings and pytest suites
ci/         # CI shell scripts
conda/      # Conda recipes / environments
docs/       # Sphinx docs
```

## PR Requirements

- All tests must pass (CI + local).
- Pre-commit must pass.
- Add tests for new functionality (both gtest and pytest where applicable).
- Update documentation for any API change.
- Sign your commits (`git commit -s`); see
  [CONTRIBUTING.md](CONTRIBUTING.md).

## Common Patterns

### CUDA error handling

```cpp
#include <rapidsmpf/error.hpp>

RAPIDSMPF_CUDA_TRY(cudaMemsetAsync(ptr, 0, size, stream.value()));
RAPIDSMPF_CUDA_TRY_ALLOC(cudaMallocAsync(&p, n, stream.value()), n);
RAPIDSMPF_EXPECTS(ptr != nullptr, "ptr must not be null");
```

### Async H2D / D2H copies

Always go through `rapidsmpf::cuda_memcpy_async` — never call
`cudaMemcpyAsync` directly:

```cpp
#include <rapidsmpf/memory/cuda_memcpy_async.hpp>

RAPIDSMPF_CUDA_TRY(rapidsmpf::cuda_memcpy_async(dst, src, n, stream));
```

### Stream-ordered code

```cpp
#include <rmm/cuda_stream_view.hpp>

void run(rmm::cuda_stream_view stream, /* ... */) {
  // All async work uses `stream`; no implicit default stream.
}
```

### NVTX annotation

```cpp
#include <rapidsmpf/nvtx.hpp>

void my_func() {
  RAPIDSMPF_NVTX_FUNC_RANGE();   // Annotate this function in NVTX
  // ...
}
```

### Cython: GIL release + exception propagation

```cython
# In a .pxd file:
cdef extern from "rapidsmpf/shuffler/shuffler.hpp" namespace "rapidsmpf::shuffler" nogil:
    cdef cppclass Shuffler:
        void insert(...) except +

# In a .pyx file, release the GIL around long-running C++ calls:
with nogil:
    self._handle.insert(...)
```

### Taking ownership of C++ buffers from Python

Always pass a Python-controlled memory resource so its lifetime is tied
to the buffer:

```python
import rmm
from rmm._lib.device_buffer import DeviceBuffer

mr = rmm.mr.get_current_device_resource()  # Python-controlled handle
buf = DeviceBuffer.c_from_unique_ptr(allocate_in_cpp(size, mr.get_mr()), mr=mr)
```

## Key Files Reference

| Purpose | Location |
|---------|----------|
| Top-level build script | [`build.sh`](build.sh) |
| C++ CMake configuration | [`cpp/CMakeLists.txt`](cpp/CMakeLists.txt) |
| C++ public headers | [`cpp/include/rapidsmpf/`](cpp/include/rapidsmpf/) |
| Error / CUDA-try macros | [`cpp/include/rapidsmpf/error.hpp`](cpp/include/rapidsmpf/error.hpp) |
| `cuda_memcpy_async` helper | [`cpp/include/rapidsmpf/memory/cuda_memcpy_async.hpp`](cpp/include/rapidsmpf/memory/cuda_memcpy_async.hpp) |
| NVTX macros | [`cpp/include/rapidsmpf/nvtx.hpp`](cpp/include/rapidsmpf/nvtx.hpp) |
| RMM resource adaptor | [`cpp/include/rapidsmpf/rmm_resource_adaptor.hpp`](cpp/include/rapidsmpf/rmm_resource_adaptor.hpp) |
| Shuffler | [`cpp/include/rapidsmpf/shuffler/`](cpp/include/rapidsmpf/shuffler/) |
| Communicators | [`cpp/include/rapidsmpf/communicator/`](cpp/include/rapidsmpf/communicator/) |
| Memory / spill | [`cpp/include/rapidsmpf/memory/`](cpp/include/rapidsmpf/memory/) |
| C++ tests | [`cpp/tests/`](cpp/tests/) |
| Python package | [`python/rapidsmpf/rapidsmpf/`](python/rapidsmpf/rapidsmpf/) |
| Python tests | [`python/rapidsmpf/rapidsmpf/tests/`](python/rapidsmpf/rapidsmpf/tests/) |
| Pre-commit config | [`.pre-commit-config.yaml`](.pre-commit-config.yaml) |
| CI scripts | [`ci/`](ci/) |
| Local CI reproduction | [`.agents/skills/reproduce-ci-locally/SKILL.md`](.agents/skills/reproduce-ci-locally/SKILL.md) |

## Resources

- **Documentation**: <https://docs.rapids.ai/api/rapidsmpf/nightly/>
- **C++ API**: <https://docs.rapids.ai/api/librapidsmpf/nightly/>
- **GitHub Issues**: <https://github.com/rapidsai/rapidsmpf/issues>
- **CONTRIBUTING**: [CONTRIBUTING.md](CONTRIBUTING.md)
