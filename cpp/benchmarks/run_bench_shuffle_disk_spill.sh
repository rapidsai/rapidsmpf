#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

repo_root="$(git -C "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)"
cd "${repo_root}"

# executor
executor=mpirun # rrun
executor_args=()
nranks=4
bench=cpp/build/benchmarks/bench_shuffle

# RAPIDSMPF env
# disk_spill_dir=/raid/nperera/spilldir
disk_spill_dir=/tmp/spilldir

# bench_shuffle
communicator=ucxx
payload_size=$((1 << 20))
insertion_batches=128
output_partitions_per_rank=2
memory_resource=async
runs=1
warmup_runs=0
discard_output=true
disable_extraction=true
unaccounted_input=true

# Pinned input is outside device accounting, so retain one receive payload.
spill_device_limit=$((payload_size))
spill_host_limit=0B
pinned_memory=false

bench_args=(
    -C "${communicator}"
    -n "${payload_size}"
    -p "${insertion_batches}"
    -o "${output_partitions_per_rank}"
    -m "${memory_resource}"
    -r "${runs}"
    -w "${warmup_runs}"
)
if [[ "${discard_output}" == true ]]; then
    bench_args+=(-s)
fi
if [[ "${disable_extraction}" == true ]]; then
    bench_args+=(-d)
fi
if [[ "${unaccounted_input}" == true ]]; then
    bench_args+=(-u)
fi

${executor} -n "${nranks}" \
    -x RAPIDSMPF_DISK_SPILL_DIR="${disk_spill_dir}" \
    -x RAPIDSMPF_SPILL_DEVICE_LIMIT="${spill_device_limit}" \
    -x RAPIDSMPF_SPILL_HOST_LIMIT="${spill_host_limit}" \
    -x RAPIDSMPF_PINNED_MEMORY="${pinned_memory}" \
    "${executor_args[@]}" \
    "${bench}" \
    "${bench_args[@]}"
