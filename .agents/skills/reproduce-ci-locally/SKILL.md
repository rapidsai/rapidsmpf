---
name: reproduce-ci-locally
description: >-
  Reproduce RAPIDS CI builds and tests locally using Docker containers that
  mirror the CI environment. Use when the user asks to reproduce CI, run CI
  locally, debug a CI failure, or simulate a CI job on their machine.
---

# Reproduce CI Locally

RAPIDS CI jobs are shell scripts executed inside
[CI containers](https://github.com/rapidsai/ci-imgs). You can reproduce them
locally by running the same container and scripts.

Reference: <https://docs.rapids.ai/resources/reproducing-ci>

## Prerequisites

- Docker installed and working (`docker run --rm hello-world`)
- For test jobs: NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (`--gpus` flag support)
- For artifact downloads: a `GH_TOKEN` (GitHub personal access token with `repo` scope)

## Information to gather from the user

Before running anything, collect these values (ask for anything not provided):

| Parameter | Description | Examples |
|-----------|-------------|----------|
| **CI script** | Which `ci/*.sh` script to run | `ci/build_cpp.sh`, `ci/test_python.sh` |
| **Container image** | The exact CI container image tag (see below) | `rapidsai/ci-conda:cuda12.8.1-ubuntu24.04-py3.12-26.06-latest` |
| **Build type** | One of `pull-request`, `nightly`, `branch` | `pull-request` |
| **Ref name** | PR number, branch, or nightly branch | `pull-request/123`, `main`, `release/26.06` |
| **Nightly date** | Date for nightly builds (YYYY-MM-DD) | `2025-12-12` (only for `nightly`) |

### Identifying the container image

CI jobs run across a matrix of container images that vary by CUDA version,
OS, Python version, and architecture. When the user asks to reproduce a CI job,
you should look up all the matrix variants from the CI run (see below) and
determine which ones are compatible with the host machine.

#### CUDA Enhanced Compatibility (CEC)

NVIDIA supports **CUDA Enhanced Compatibility**: a container image built for
CUDA X.Y can run on any host driver that supports CUDA major version X,
regardless of the minor version. As a rule of thumb, **any image from the same
CUDA major version as the host driver works**. For example, a host with driver
535 (which ships with CUDA 12.2) can run images for CUDA 12.2, 12.5, 12.8,
12.9, etc. — but **not** CUDA 13.x images.

To determine the host's CUDA major version, check the driver version with
`nvidia-smi` and map it to the CUDA major version:

| Driver series | CUDA major version |
|---------------|--------------------|
| 525–560+      | 12                 |
| 570+          | 13                 |

#### Choosing an image when multiple are compatible

When there are **multiple compatible images** for the same CI job (e.g.,
CUDA 12.2.2 and CUDA 12.9.1 variants both compatible via CEC), **always
present all compatible options to the user and ask them to choose**. Do not
silently pick one. List each option with its key attributes (CUDA version,
OS, Python version, GPU type used in CI) so the user can make an informed
decision.

Only auto-select an image when exactly one variant is compatible with the host.

#### Finding the images from CI job logs

If the user provides the image tag directly, use it. Otherwise, look up the
images from the CI run:

1. Query the GitHub API for the jobs in the workflow run (see the
   `actions/runs/{run_id}/jobs` endpoint).
2. For each relevant job, download its logs (requires `GH_TOKEN` with `repo`
   scope) and look for the `docker pull` line in the **Initialize Containers**
   / **Starting job container** section — it contains the full image tag
   (e.g., `rapidsai/ci-conda:26.06-cuda12.9.1-ubuntu22.04-py3.11`).
3. Alternatively, for jobs in `.github/workflows/` that set `container_image`
   explicitly, read the value from the workflow file. However, most jobs use
   shared workflows (e.g., `conda-cpp-tests.yaml`) which select the image
   from a build matrix — in that case the CI job logs are the only reliable
   source.

### Available CI scripts in this project

**Conda builds (no GPU needed):**
- `ci/build_cpp.sh` — Build C++ conda package
- `ci/build_python.sh` — Build Python conda package (needs C++ artifacts)
- `ci/build_docs.sh` — Build documentation (needs C++ and Python artifacts)

**Conda tests (GPU needed):**
- `ci/test_cpp.sh` — Test C++ conda package
- `ci/test_python.sh` — Test Python conda package
- `ci/test_cpp_memcheck.sh` — Test C++ with compute-sanitizer memcheck

**Wheel builds (no GPU needed):**
- `ci/build_wheel_librapidsmpf.sh` — Build librapidsmpf wheel
- `ci/build_wheel_rapidsmpf.sh` — Build rapidsmpf wheel
- `ci/build_wheel_singlecomm.sh` — Build rapidsmpf single-comm wheel (no MPI/UCXX)

**Wheel tests (GPU needed):**
- `ci/test_wheel.sh` — Test rapidsmpf wheel

**Linters / checks (no GPU needed):**
- `ci/check_style.sh` — Pre-commit style checks
- `ci/cpp_linters.sh` — C++ clang-tidy linting

## Running a CI job locally

### Step 1: Launch the container

For **build** jobs (no GPU required):

```bash
docker run \
  --rm \
  --pull=always \
  --volume $PWD:/repo \
  --workdir /repo \
  <CONTAINER_IMAGE>
```

For **test** jobs (GPU required):

```bash
docker run \
  --rm \
  --gpus all \
  --pull=always \
  --cap-add CAP_SYS_PTRACE \
  --shm-size=8g \
  --ulimit nofile=1000000:1000000 \
  --volume $PWD:/repo \
  --workdir /repo \
  <CONTAINER_IMAGE>
```

The `--cap-add`, `--shm-size`, and `--ulimit` flags match what CI uses for
RapidsMPF test jobs (see `container-options` in the workflow files).

**Note on `-it` flag:** Do NOT use `-it` (or `-t`) when running a command
non-interactively via `bash -c "..."`. Docker requires a real TTY for `-t` and
will fail with "the input device is not a TTY" in background/scripted execution.
Only add `-it` when launching an interactive shell for manual exploration.

### Step 2: Set environment variables for artifact downloads

Test scripts download build artifacts from GitHub Actions. Inside the container,
set these variables so the download commands don't prompt interactively:

```bash
export RAPIDS_BUILD_TYPE=pull-request   # or "nightly" or "branch"
export RAPIDS_REPOSITORY=rapidsai/rapidsmpf

# For pull-request builds:
export RAPIDS_REF_NAME=pull-request/123

# For nightly builds:
export RAPIDS_REF_NAME=main
export RAPIDS_NIGHTLY_DATE=2025-12-12

# For branch builds:
export RAPIDS_REF_NAME=release/26.06
```

#### Setting `RAPIDS_SHA` (required for test jobs)

The `rapids-download-conda-from-github` / `rapids-download-from-github` scripts
need the exact commit SHA that produced the build artifacts. This must match the
SHA used by the CI build job — **not** the current tip of the branch. For
example, if nightly ran on commit `abcdef0` but a later commit `abcdef1` was
pushed to `main` afterward, the artifacts only exist for `abcdef0`. Using
`abcdef1` will fail to find them.

When the local repo is a **fork** (i.e., `origin` points to a user fork rather
than `rapidsai/rapidsmpf`), the scripts also cannot determine the SHA
automatically and will fail with:

> There was a problem acquiring the HEAD commit sha from the current directory.

**Always set `RAPIDS_SHA` explicitly** to the SHA from the CI run you want to
reproduce. There are three ways to look it up:

**Option A — GitHub REST API (no extra tools needed):**

For nightly builds, query the `build.yaml` workflow runs:

```bash
curl -s "https://api.github.com/repos/rapidsai/rapidsmpf/actions/workflows/build.yaml/runs?per_page=5" \
  | python3 -c "import sys,json; runs=json.load(sys.stdin)['workflow_runs']; print('\n'.join(f\"{r['head_sha'][:12]}  {r['created_at']}\" for r in runs))"
```

For pull-request builds, query the `pr.yaml` workflow runs for a specific PR
branch:

```bash
curl -s "https://api.github.com/repos/rapidsai/rapidsmpf/actions/workflows/pr.yaml/runs?branch=pull-request/123&per_page=5" \
  | python3 -c "import sys,json; runs=json.load(sys.stdin)['workflow_runs']; print('\n'.join(f\"{r['head_sha'][:12]}  {r['created_at']}\" for r in runs))"
```

Pick the `head_sha` from the run matching your target date.

**Option B — `gh` CLI (if installed):**

```bash
gh run list --repo rapidsai/rapidsmpf --workflow build.yaml --json headSha,createdAt --limit 5
```

**Option C — GitHub web UI:**

Browse to <https://github.com/rapidsai/rapidsmpf/actions/workflows/build.yaml>,
click the run for the target date, and note the commit SHA shown at the top.

After obtaining the SHA, **check out that exact commit locally** so the repo
contents match the artifacts, then set the variable:

```bash
git fetch <upstream-remote> <branch>
git checkout <sha>
export RAPIDS_SHA=<sha>
```

Where `<upstream-remote>` is the remote pointing to `rapidsai/rapidsmpf`
(commonly `upstream`). For example, for a nightly build on `main`:

```bash
RAPIDS_SHA=$(curl -s "https://api.github.com/repos/rapidsai/rapidsmpf/actions/workflows/build.yaml/runs?per_page=1" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['workflow_runs'][0]['head_sha'])")
git fetch upstream main
git checkout $RAPIDS_SHA
export RAPIDS_SHA
```

### Step 3: Authenticate with GitHub (for test jobs)

Test scripts use `rapids-download-conda-from-github` /
`rapids-download-from-github` which require GitHub authentication via `GH_TOKEN`.

**Option A — pass the token directly:**

```bash
docker run \
  ...
  --env "GH_TOKEN=<your-token>" \
  ...
```

**Option B — use an env file:**

Create a `.env` file with `GH_TOKEN=<your-token>` and mount it:

```bash
docker run \
  ...
  --env-file "$(pwd)/.env" \
  ...
```

On shared machines, passing tokens via `--env` may expose them in `ps` output.
Prefer `--env-file` in that case.

Ask the user for their `GH_TOKEN` if not already set in the environment.

### Step 4: Run the CI script

```bash
./ci/build_cpp.sh        # or whichever script you want to reproduce
```

## Full build-then-test workflow (local artifacts)

CI builds and tests run on separate machines. To do a complete local
build+test cycle without downloading artifacts from GitHub, redirect the
channel variables to use locally-built packages:

```bash
sed -ri '/rapids-download.*from-github/ s/_CHANNEL=.*/_CHANNEL=${RAPIDS_CONDA_BLD_OUTPUT_DIR}/' ci/*.sh

./ci/build_cpp.sh
./ci/build_python.sh
./ci/test_cpp.sh
./ci/test_python.sh
```

**Important:** this `sed` modifies CI scripts in your working tree. Either do
this on a throwaway branch or revert the changes afterward
(`git checkout -- ci/`).

## Common issues

### Missing git tags

Some builds need version info from git tags. If you see errors like
`'GIT_DESCRIBE_NUMBER' is undefined`, fetch tags from upstream:

```bash
git fetch git@github.com:rapidsai/rapidsmpf.git --tags
```

### GPU driver version mismatch

CI tests may run on different GPU driver versions. If local test results differ
from CI, compare `nvidia-smi` output between your machine and the CI job logs.

### Build artifacts cannot be uploaded

Locally-built artifacts cannot be uploaded to CI artifact storage. Fix build
failures locally, push to PR, and let CI produce the artifacts for test jobs.

## Putting it all together — example

The agent should construct and run the full `docker run` command based on user
input. Here is an example reproducing a failing `test_python.sh` from PR #123:

```bash
docker run \
  --rm \
  --gpus all \
  --pull=always \
  --cap-add CAP_SYS_PTRACE \
  --shm-size=8g \
  --ulimit nofile=1000000:1000000 \
  --volume $PWD:/repo \
  --workdir /repo \
  --env-file "$(pwd)/.env" \
  --env RAPIDS_BUILD_TYPE=pull-request \
  --env RAPIDS_REPOSITORY=rapidsai/rapidsmpf \
  --env RAPIDS_REF_NAME=pull-request/123 \
  --env RAPIDS_SHA=<commit-sha> \
  <CONTAINER_IMAGE> \
  bash -c "./ci/test_python.sh"
```

Replace `<CONTAINER_IMAGE>` with the exact image from the CI job logs (see
"Identifying the container image" above) and `<commit-sha>` with the full SHA
from the CI run (see "Setting `RAPIDS_SHA`" above).

To get an interactive shell instead, drop `bash -c "..."` and add `-it`:

```bash
docker run \
  --rm \
  --gpus all \
  --pull=always \
  --cap-add CAP_SYS_PTRACE \
  --shm-size=8g \
  --ulimit nofile=1000000:1000000 \
  --volume $PWD:/repo \
  --workdir /repo \
  --env-file "$(pwd)/.env" \
  --env RAPIDS_BUILD_TYPE=pull-request \
  --env RAPIDS_REPOSITORY=rapidsai/rapidsmpf \
  --env RAPIDS_REF_NAME=pull-request/123 \
  --env RAPIDS_SHA=<commit-sha> \
  -it <CONTAINER_IMAGE>
```
