#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

set -euo pipefail

# Support invoking run_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/rapidsmp

# TODO: call pytest. For now, we just check if the module was built
python -c "import rapidsmp;print(rapidsmp)"
# pytest --cache-clear --verbose "$@" tests
