#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Sweep shuffler send order policies (RAPIDSMPF_SHUFFLER_SEND_ORDER_POLICY) over
# scenarios, device memory limits and payload sizes using bench_shuffle.
#
# Usage: run_send_policy_sweep.sh <build-dir> <out-dir>
#
# Environment knobs (defaults in parentheses):
#   NRANKS       number of ranks (4)
#   LAUNCHER     mpirun or rrun (mpirun)
#   MPIRUN       mpirun executable (mpirun)
#   RRUN         rrun executable (<build-dir>/tools/rrun)
#   GPUS         rrun GPU list, e.g. 0,1,2,3 (0 repeated NRANKS times)
#   POLICIES     ("global rank pid none")
#   LIMITS       device limit factors of the local input size ("unlimited 2 1 0.5")
#   SCENARIOS    ("S0 S1 S2 S4")
#   SIZES        "<payload>x<batches>x<partitions per rank>" ("1048576x16x2 8388608x8x2")
#   DELAY_US     max batch delay for S1/S2/S4 (100000). Make it comparable to the
#                shuffle time, otherwise chunks are ready before the progress thread
#                looks at them and the policies never differ.
#   RUNS/WARMUPS (5/1)
#   RMM_MR       (async)
#   TIMEOUT      per cell timeout in seconds (900)
#   PRELOAD      optional library to LD_PRELOAD (e.g. a freshly built librapidsmpf.so)
#
# Scenarios:
#   S0  pre-generated device input, all chunks ready (control)
#   S1  batches generated on their own streams, reverse delay (first batch ready last)
#   S2  as S1, random delay
#   S4  as S1, but batch b only covers partitions p with p % 2 == b % 2

set -u

BUILD_DIR=$(realpath "${1:?build dir required}")
OUT_DIR=${2:?output dir required}
mkdir -p "${OUT_DIR}/logs"
OUT_DIR=$(realpath "${OUT_DIR}")

NRANKS=${NRANKS:-4}
LAUNCHER=${LAUNCHER:-mpirun}
MPIRUN=${MPIRUN:-mpirun}
RRUN=${RRUN:-${BUILD_DIR}/tools/rrun}
GPUS=${GPUS:-$(printf '0,%.0s' $(seq "${NRANKS}") | sed 's/,$//')}
POLICIES=${POLICIES:-"global rank pid none"}
LIMITS=${LIMITS:-"unlimited 2 1 0.5"}
SCENARIOS=${SCENARIOS:-"S0 S1 S2 S4"}
SIZES=${SIZES:-"1048576x16x2 8388608x8x2"}
DELAY_US=${DELAY_US:-100000}
RUNS=${RUNS:-5}
WARMUPS=${WARMUPS:-1}
RMM_MR=${RMM_MR:-async}
TIMEOUT=${TIMEOUT:-900}
PRELOAD=${PRELOAD:-}
BENCH=${BUILD_DIR}/benchmarks/bench_shuffle
CSV=${OUT_DIR}/results.csv


scenario_args() {
    case "$1" in
        S0) echo "" ;;
        S1) echo "-g -d ${DELAY_US}:reverse" ;;
        S2) echo "-g -d ${DELAY_US}:random" ;;
        S4) echo "-g -d ${DELAY_US}:reverse -k 2" ;;
        *) echo "unknown scenario $1" >&2; exit 1 ;;
    esac
}

run_cell() {
    local policy=$1 log=$2
    shift 2
    local env_vars=(
        "RAPIDSMPF_SHUFFLER_SEND_ORDER_POLICY=${policy}"
    )
    if [[ -n "${PRELOAD}" ]]; then
        env_vars+=("LD_PRELOAD=${PRELOAD}")
    fi
    if [[ "${LAUNCHER}" == "rrun" ]]; then
        timeout "${TIMEOUT}" env "${env_vars[@]}" \
            "${RRUN}" -n "${NRANKS}" -g "${GPUS}" "${BENCH}" -C ucxx "$@" > "${log}" 2>&1
    else
        local mpi_env=()
        for kv in "${env_vars[@]}"; do
            mpi_env+=(-x "${kv}")
        done
        timeout "${TIMEOUT}" "${MPIRUN}" --map-by node --bind-to none -np "${NRANKS}" \
            "${mpi_env[@]}" "${BENCH}" -C ucxx "$@" > "${log}" 2>&1
    fi
}

echo "size,scenario,limit,policy,rc,max_mean_elapsed_s,local_input_bytes,device_limit,peak_device_bytes,spilled_bytes,not_ready,blocked,oom,log" > "${CSV}"

for size in ${SIZES}; do
    IFS=x read -r payload batches parts <<< "${size}"
    for scenario in ${SCENARIOS}; do
        for limit in ${LIMITS}; do
            for policy in ${POLICIES}; do
                log="${OUT_DIR}/logs/${size}_${scenario}_L${limit}_${policy}.log"
                # shellcheck disable=SC2046
                run_cell "${policy}" "${log}" -m "${RMM_MR}" -n "${payload}" \
                    -p "${batches}" -o "${parts}" -L "${limit}" -r "${RUNS}" \
                    -w "${WARMUPS}" $(scenario_args "${scenario}")
                rc=$?
                row=$(python3 - "${log}" <<'EOF'
import re, sys

text = open(sys.argv[1], errors="replace").read()
units = {"B": 1, "KiB": 2**10, "MiB": 2**20, "GiB": 2**30, "TiB": 2**40}

def nbytes(value, unit):
    return float(value) * units[unit]

elapsed = [float(x) for x in re.findall(r"RESULT .*?mean_elapsed_s=([0-9.e+-]+)", text)]
m = re.search(r"RESULT .*?local_input_bytes=(\d+) .*?device_limit=(\S+)", text)
local_input, device_limit = (m.group(1), m.group(2)) if m else ("", "")
# Global peak of all allocations through the RMM adaptor, per rank.
peaks = [
    nbytes(v, u)
    for v, u in re.findall(
        r"^\s+\d+\s+\S+ \S+\s+([0-9.]+) (\w+)\s+\S+ \S+\s+\S+ \S+\s+main \(all", text, re.M
    )
]
spilled = sum(
    nbytes(v, u)
    for v, u in re.findall(r"- copy-device-to-[^:]+:\s+([0-9.]+) (\w+)", text)
)

def counter(name):
    return sum(int(v) for v in re.findall(rf"- {name}:\s+(\d+)", text))

print(",".join(str(x) for x in [
    max(elapsed) if len(elapsed) else "",
    local_input,
    device_limit,
    int(max(peaks)) if peaks else "",
    int(spilled),
    counter("shuffle-send-not-ready"),
    counter("shuffle-send-blocked"),
    int(bool(re.search(r"out of memory|out_of_memory|reserve_or_fail|bad_alloc", text, re.I))),
]))
EOF
)
                echo "${size},${scenario},${limit},${policy},${rc},${row},${log}" >> "${CSV}"
                echo "${size} ${scenario} L=${limit} ${policy}: rc=${rc} ${row}"
            done
        done
    done
done

echo "Results: ${CSV}"
