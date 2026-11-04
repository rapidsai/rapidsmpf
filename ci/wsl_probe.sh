#!/bin/bash
# TEMP: WSL diagnostics for ShufflerManyStreams slowdown.
set -x
nproc
free -g
cat /proc/cpuinfo | grep "model name" | head -1
uname -a
cd "${CONDA_PREFIX}/bin/tests/librapidsmpf/" || exit 1
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 OMPI_MCA_opal_cuda_support=1
F="--gtest_filter=ShufflerManyStreams.*"
run() {
    local label="$1"; shift
    echo "=== ${label}"
    timeout 300 "$@" 2>&1 | grep -E "OK \]|FAILED  \]|wait timeout|rror" | sort | uniq -c
}
for np in 1 2 5; do
    run "default np=${np}" mpirun --map-by node --bind-to none -np "${np}" ./mpi_tests "${F}"
done
run "UCX_TLS=^cuda_ipc np=5" mpirun --map-by node --bind-to none -x UCX_TLS=^cuda_ipc -np 5 ./mpi_tests "${F}"
run "pml ob1 np=5" mpirun --map-by node --bind-to none --mca pml ob1 -np 5 ./mpi_tests "${F}"
run "ucxx np=5" mpirun --map-by node --bind-to none -np 5 ./ucxx_tests "${F}"
run "default np=5 repeat" mpirun --map-by node --bind-to none -np 5 ./mpi_tests "${F}"
echo "=== bench_memory_resources"
cd "${CONDA_PREFIX}/bin/benchmarks/librapidsmpf/" && RAPIDSMPF_SMOKE_TEST_MODE=ON timeout 120 ./bench_memory_resources 2>&1 | grep -E "BM_"
exit 1
