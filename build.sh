#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

# rapidsmpf build script

# This script is used to build the component(s) in this repo from
# source, and can be called with various options to customize the
# build as needed (see the help output for details)

# Abort script on first error
set -e

NUMARGS=$#
ARGS=$*

# NOTE: ensure all dir changes are relative to the location of this
# script, and that this script resides in the repo dir!
REPODIR=$(cd "$(dirname "$0")"; pwd)

VALIDARGS="clean librapidsmpf rapidsmpf -v -g -n --pydevelop --asan --no-clang-tidy -h"
HELP="$0 [clean] [librapidsmpf] [rapidsmpf] [-v] [-g] [-n] [--cmake-args=\"<args>\"] [--asan] [--no-clang-tidy] [-h]
   clean                       - remove all existing build artifacts and configuration (start over)
   librapidsmpf                - build and install the librapidsmpf C++ code
   rapidsmpf                   - build the rapidsmpf Python package
   -v                          - verbose build mode
   -g                          - build for debug
   -n                          - no install step
   --pydevelop                 - Install Python packages in editable mode
   --cmake-args=\\\"<args>\\\" - pass arbitrary list of CMake configuration options (escape all quotes in argument)
   --asan                      - enable AddressSanitizer for C++ and Python builds
   --no-clang-tidy             - disable clang-tidy build checks (default: enabled)
   -h                          - print this text
   default action (no args) is to build and install the 'librapidsmpf' then 'rapidsmpf' targets
"
LIBRAPIDSMPF_BUILD_DIR=${LIBRAPIDSMPF_BUILD_DIR:=${REPODIR}/cpp/build}
PYRAPIDSMPF_=${REPODIR}/python/rapidsmpf/build
BUILD_DIRS="${LIBRAPIDSMPF_BUILD_DIR} ${PYRAPIDSMPF_}"

# Set defaults for vars modified by flags to this script
VERBOSE_FLAG=""
BUILD_TYPE=Release
BUILD_ALL_GPU_ARCH=0
INSTALL_TARGET=install
RAN_CMAKE=0
PYTHON_ARGS_FOR_INSTALL=("-m" "pip" "install" "--no-build-isolation" "--no-deps" "--config-settings" "rapidsai.disable-cuda=true")

# Set defaults for vars that may not have been defined externally
# If INSTALL_PREFIX is not set, check PREFIX, then check
# CONDA_PREFIX, then fall back to install inside of $LIBRAPIDSMPF_BUILD_DIR
INSTALL_PREFIX=${INSTALL_PREFIX:=${PREFIX:=${CONDA_PREFIX:=$LIBRAPIDSMPF_BUILD_DIR/install}}}
PARALLEL_LEVEL=${PARALLEL_LEVEL:=$(nproc --all)}

function hasArg {
    (( NUMARGS != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

function buildAll {
    ! hasArg librapidsmpf && ! hasArg rapidsmpf
}

function cmakeArgs {
    # Check for multiple cmake args options
    if [[ $(echo "$ARGS" | { grep -Eo "\-\-cmake\-args" || true; } | wc -l ) -gt 1 ]]; then
        echo "Multiple --cmake-args options were provided, please provide only one: ${ARGS}"
        exit 1
    fi

    # Check for cmake args option
    if [[ -n $(echo "$ARGS" | { grep -E "\-\-cmake\-args" || true; } ) ]]; then
        # There are possible weird edge cases that may cause this regex filter to output nothing and fail silently
        # the true pipe will catch any weird edge cases that may happen and will cause the program to fall back
        # on the invalid option error
        EXTRA_CMAKE_ARGS=$(echo "$ARGS" | { grep -Eo "\-\-cmake\-args=\".+\"" || true; })
        if [[ -n ${EXTRA_CMAKE_ARGS} ]]; then
            # Remove the full  EXTRA_CMAKE_ARGS argument from list of args so that it passes validArgs function
            ARGS=${ARGS//$EXTRA_CMAKE_ARGS/}
            # Filter the full argument down to just the extra string that will be added to cmake call
            EXTRA_CMAKE_ARGS=$(echo "$EXTRA_CMAKE_ARGS" | grep -Eo "\".+\"" | sed -e 's/^"//' -e 's/"$//')
        fi
    fi

    read -ra EXTRA_CMAKE_ARGS <<< "$EXTRA_CMAKE_ARGS"
}


# Runs cmake if it has not been run already for build directory
# LIBRAPIDSMPF_BUILD_DIR
function ensureCMakeRan {
    mkdir -p "${LIBRAPIDSMPF_BUILD_DIR}"
    cd "${REPODIR}"/cpp
    if (( RAN_CMAKE == 0 )); then
        echo "Executing cmake for librapidsmpf..."
        CMAKE_ARGS=(-B "${LIBRAPIDSMPF_BUILD_DIR}" -S . \
              -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
              -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
              -DCMAKE_CUDA_ARCHITECTURES="${RAPIDSMPF_CMAKE_CUDA_ARCHITECTURES}")
        if hasArg --no-clang-tidy; then
            CMAKE_ARGS+=(-DRAPIDSMPF_CLANG_TIDY=OFF)
        else
            CMAKE_ARGS+=(-DRAPIDSMPF_CLANG_TIDY=ON)
        fi

        if hasArg --asan; then
            CMAKE_ARGS+=(-DRAPIDSMPF_ASAN=ON)
        fi
        CMAKE_ARGS+=("${EXTRA_CMAKE_ARGS[@]}")
        cmake "${CMAKE_ARGS[@]}"
        RAN_CMAKE=1
    fi
}

if hasArg -h || hasArg --help; then
    echo "${HELP}"
    exit 0
fi

# Check for valid usage
if (( NUMARGS != 0 )); then
    # Check for cmake args
    cmakeArgs
    for a in ${ARGS}; do
    if ! (echo " ${VALIDARGS} " | grep -q " ${a} "); then
        echo "Invalid option or formatting, check --help: ${a}"
        exit 1
    fi
    done
fi

# Process flags
if hasArg -v; then
    VERBOSE_FLAG=-v
    set -x
fi
if hasArg -g; then
    BUILD_TYPE=Debug
fi
if hasArg -n; then
    INSTALL_TARGET=""
fi

if hasArg --pydevelop; then
    PYTHON_ARGS_FOR_INSTALL+=("-e")
fi

# If clean given, run it prior to any other steps
if hasArg clean; then
    # If the dirs to clean are mounted dirs in a container, the
    # contents should be removed but the mounted dirs will remain.
    # The find removes all contents but leaves the dirs, the rmdir
    # attempts to remove the dirs but can fail safely.
    for bd in ${BUILD_DIRS}; do
        if [ -d "${bd}" ]; then
            find "${bd}" -mindepth 1 -delete
            rmdir "${bd}" || true
        fi
    done
fi

################################################################################
# Configure, build, and install librapidsmpf

if buildAll || hasArg librapidsmpf || hasArg rapidsmpf ; then
    if (( BUILD_ALL_GPU_ARCH == 0 )); then
        RAPIDSMPF_CMAKE_CUDA_ARCHITECTURES="${RAPIDSMPF_CMAKE_CUDA_ARCHITECTURES:-NATIVE}"
        if [[ "$RAPIDSMPF_CMAKE_CUDA_ARCHITECTURES" == "NATIVE" ]]; then
            echo "Building for the architecture of the GPU in the system..."
        else
            echo "Building for the GPU architecture(s) $RAPIDSMPF_CMAKE_CUDA_ARCHITECTURES ..."
        fi
    else
        RAPIDSMPF_CMAKE_CUDA_ARCHITECTURES="RAPIDS"
        echo "Building for *ALL* supported GPU architectures..."
    fi
fi

if (( NUMARGS == 0 )) || hasArg librapidsmpf; then
    ensureCMakeRan
    echo "building librapidsmpf..."
    cmake --build "${LIBRAPIDSMPF_BUILD_DIR}" -j"${PARALLEL_LEVEL}" ${VERBOSE_FLAG}
    if [[ ${INSTALL_TARGET} != "" ]]; then
        echo "installing librapidsmpf..."
        cmake --build "${LIBRAPIDSMPF_BUILD_DIR}" --target install ${VERBOSE_FLAG}
    fi
fi

# Build and install the rapidsmpf Python package
if (( NUMARGS == 0 )) || hasArg rapidsmpf; then
    echo "building rapidsmpf..."
    cd "${REPODIR}"/python/rapidsmpf

    SKBUILD_CMAKE_ARGS="-DCMAKE_PREFIX_PATH=${INSTALL_PREFIX};-DCMAKE_LIBRARY_PATH=${LIBRAPIDSMPF_BUILD_DIR}"

    if hasArg --asan; then
        SKBUILD_CMAKE_ARGS="${SKBUILD_CMAKE_ARGS};-DRAPIDSMPF_PYTHON_ASAN=ON"
    fi

    if [[ -n "${EXTRA_CMAKE_ARGS[*]}" ]]; then
        SKBUILD_CMAKE_ARGS="${SKBUILD_CMAKE_ARGS};${EXTRA_CMAKE_ARGS[*]// /;}"
    fi

    SKBUILD_CMAKE_ARGS="${SKBUILD_CMAKE_ARGS}" \
        python "${PYTHON_ARGS_FOR_INSTALL[@]}" ${VERBOSE_FLAG} .
fi
