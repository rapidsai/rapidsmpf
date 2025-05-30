#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
################################################################################
# rapidsmpf version updater
################################################################################

## Usage
# bash update-version.sh <new_version>

# Format is YY.MM.PP - no leading 'v' or trailing 'a'
NEXT_FULL_TAG=$1

# Get current version
CURRENT_TAG=$(git tag --merged HEAD | grep -xE '^v.*' | sort --version-sort | tail -n 1 | tr -d 'v')

#Get <major>.<minor> for next version
NEXT_MAJOR=$(echo "$NEXT_FULL_TAG" | awk '{split($0, a, "."); print a[1]}')
NEXT_MINOR=$(echo "$NEXT_FULL_TAG" | awk '{split($0, a, "."); print a[2]}')
NEXT_SHORT_TAG=${NEXT_MAJOR}.${NEXT_MINOR}
NEXT_SHORT_TAG_PEP440=$(python -c "from packaging.version import Version; print(Version('${NEXT_SHORT_TAG}'))")

NEXT_UCXX_SHORT_TAG="$(curl -sL https://version.gpuci.io/rapids/"${NEXT_SHORT_TAG}")"
NEXT_UCXX_SHORT_TAG_PEP440=$(python -c "from packaging.version import Version; print(Version('${NEXT_UCXX_SHORT_TAG}'))")

echo "Preparing release $CURRENT_TAG => $NEXT_FULL_TAG"

# Inplace sed replace; workaround for Linux and Mac
function sed_runner() {
  sed -i.bak ''"$1"'' "$2" && rm -f "${2}".bak
}

# Centralized version file update
echo "${NEXT_FULL_TAG}" | tr -d '"' >VERSION

# Bump testing dependencies
sed_runner "s/ucxx==.*/ucxx==${NEXT_UCXX_SHORT_TAG_PEP440}.*,>=0.0.0a0/g" dependencies.yaml

DEPENDENCIES=(
  cudf
  dask-cuda
  dask-cudf
  libcudf
  pylibcudf
  rapidsmpf
  rmm
)
for DEP in "${DEPENDENCIES[@]}"; do
  for FILE in dependencies.yaml conda/environments/*.yaml; do
    sed_runner "/-.* ${DEP}\(-cu[[:digit:]]\{2\}\)\{0,1\}==/ s/==.*/==${NEXT_SHORT_TAG_PEP440}.*,>=0.0.0a0/g" "${FILE}"
  done
  sed_runner "/\"${DEP}==/ s/==.*\"/==${NEXT_SHORT_TAG_PEP440}.*,>=0.0.0a0\"/g" python/rapidsmpf/pyproject.toml
done

UCX_DEPENDENCIES=(
  libucxx
)
for DEP in "${UCX_DEPENDENCIES[@]}"; do
  for FILE in dependencies.yaml conda/environments/*.yaml; do
    sed_runner "/-.* ${DEP}\(-cu[[:digit:]]\{2\}\)\{0,1\}==/ s/==.*/==${NEXT_UCXX_SHORT_TAG_PEP440}.*,>=0.0.0a0/g" "${FILE}"
  done
  sed_runner "/\"${DEP}==/ s/==.*\"/==${NEXT_UCXX_SHORT_TAG_PEP440}.*,>=0.0.0a0\"/g" python/rapidsmpf/pyproject.toml
done

# RAPIDS UCX version
for FILE in conda/recipes/*/conda_build_config.yaml; do
  sed_runner "/^ucx_py_version:$/ {n;s/.*/  - \"${NEXT_UCXX_SHORT_TAG_PEP440}.*\"/}" "${FILE}"
  sed_runner "/^ucxx_version:$/ {n;s/.*/  - \"${NEXT_UCXX_SHORT_TAG_PEP440}.*\"/}" "${FILE}"
done

# rapids-cmake version
sed_runner 's/'"set(rapids-cmake-version.*"'/'"set(rapids-cmake-version ${NEXT_RAPIDS_SHORT_TAG})"'/g' cmake/RAPIDS.cmake

# CI files
for FILE in .github/workflows/*.yaml; do
  sed_runner "/shared-workflows/ s/@.*/@branch-${NEXT_SHORT_TAG}/g" "${FILE}"
done
