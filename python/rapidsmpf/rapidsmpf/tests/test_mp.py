# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the multiprocessing-based SPMD pool integration."""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import numpy as np
import pytest

import cudf
import pylibcudf
from rmm.pylibrmm.stream import DEFAULT_STREAM

from rapidsmpf.coll.allgather import AllGather
from rapidsmpf.config import Options
from rapidsmpf.integrations.cudf.partition import (
    partition_and_pack,
    unpack_and_concat,
    unspill_partitions,
)
from rapidsmpf.integrations.mp import MultiprocessingPool, get_worker_context
from rapidsmpf.memory.packed_data import PackedData
from rapidsmpf.rrun.rrun import ResourceBinding, check_binding
from rapidsmpf.shuffler import Shuffler
from rapidsmpf.testing import assert_eq
from rapidsmpf.utils.cudf import cudf_to_pylibcudf_table, pylibcudf_to_cudf_dataframe

if TYPE_CHECKING:
    from collections.abc import Generator


def _get_nranks_if_spawned_by_mpi() -> int:
    """Return the MPI world size if running under an MPI launcher, else -1."""
    mpi_env_vars = [
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "MPI_WORLD_SIZE",
        "MV2_COMM_WORLD_SIZE",
    ]
    for var in mpi_env_vars:
        if var in os.environ:
            return int(os.environ[var])
    return -1


pytestmark = pytest.mark.skipif(
    _get_nranks_if_spawned_by_mpi() > 1,
    reason="Multiprocessing pool tests must not run under an MPI launcher",
)


# ---------------------------------------------------------------------------
# Module-level helper functions executed on workers via pool.run().
# These must be defined at module scope so cloudpickle can serialise them
# either by reference (when imported) or by value (lambdas / closures below).
# ---------------------------------------------------------------------------


def _worker_comm_nranks() -> int:
    """Return comm.nranks from the calling worker's context."""
    return get_worker_context().comm.nranks


def _worker_comm_rank() -> int:
    """Return comm.rank from the calling worker's context."""
    return get_worker_context().comm.rank


def _worker_comm_str() -> str:
    """Return comm.get_str() from the calling worker's context."""
    return get_worker_context().comm.get_str()


def _worker_check_binding() -> ResourceBinding:
    """Return rrun.check_binding() result from the calling worker."""
    return check_binding()


def _worker_raise() -> None:
    """Always raise — used to test error propagation."""
    raise ValueError("deliberate worker error")


def _worker_raise_on_even_ranks() -> None:
    """Raise on even-numbered ranks — used to test partial error propagation."""
    rank = get_worker_context().comm.rank
    if rank % 2 == 0:
        raise ValueError(f"deliberate error on rank {rank}")


def _worker_cuda_visible_devices() -> str | None:
    """Return CUDA_VISIBLE_DEVICES from the calling worker's environment."""
    return os.environ.get("CUDA_VISIBLE_DEVICES")


def _worker_run_shuffle(total_num_partitions: int, num_rows: int) -> int:
    """Run a distributed shuffle on this worker and return the extracted row count."""
    ctx = get_worker_context()
    comm = ctx.comm
    br = ctx.br

    np.random.seed(42)
    df = cudf.DataFrame(
        {
            "a": cudf.Series(range(num_rows), dtype="int32"),
            "b": cudf.Series(np.random.randint(0, 1000, num_rows), dtype="int32"),
        }
    )
    columns_to_hash = (df.columns.get_loc("b"),)

    shuffler = Shuffler(comm, op_id=0, total_num_partitions=total_num_partitions, br=br)

    stride = math.ceil(num_rows / comm.nranks)
    local_df = df.iloc[comm.rank * stride : (comm.rank + 1) * stride]
    packed_inputs = partition_and_pack(
        cudf_to_pylibcudf_table(local_df),
        columns_to_hash=columns_to_hash,
        num_partitions=total_num_partitions,
        br=br,
        stream=DEFAULT_STREAM,
    )
    shuffler.insert_chunks(packed_inputs)
    shuffler.insert_finished()
    shuffler.wait()

    extracted_rows = 0
    for partition_id in shuffler.local_partitions():
        packed_chunks = shuffler.extract(partition_id)
        partition = unpack_and_concat(
            unspill_partitions(packed_chunks, br=br, allow_overbooking=True),
            br=br,
            stream=DEFAULT_STREAM,
        )
        extracted_rows += len(pylibcudf_to_cudf_dataframe(partition))

    shuffler.shutdown()
    return extracted_rows


def _worker_allgather_and_verify(run_id: int, rows_per_rank: int) -> None:
    """Each rank contributes a unique table; allgather; verify all N tables in-worker."""
    ctx = get_worker_context()
    comm = ctx.comm
    br = ctx.br

    # Each rank creates a table whose "rank" column identifies the source.
    local_df = cudf.DataFrame(
        {
            "rank": cudf.Series([comm.rank] * rows_per_rank, dtype="int32"),
            "value": cudf.Series(
                range(run_id * rows_per_rank, (run_id + 1) * rows_per_rank),
                dtype="int32",
            ),
        }
    )
    # Pack the local table into a single PackedData using cudf's contiguous pack.
    packed_local = PackedData.from_cudf_packed_columns(
        pylibcudf.contiguous_split.pack(cudf_to_pylibcudf_table(local_df)),
        DEFAULT_STREAM,
        br,
    )

    ag = AllGather(comm, op_id=0, br=br)
    ag.insert(0, packed_local)
    ag.insert_finished()
    gathered = ag.wait_and_extract(ordered=True)

    # ordered=True → one PackedData per rank, concatenate all at once.
    values = list(range(run_id * rows_per_rank, (run_id + 1) * rows_per_rank))
    expected_df = cudf.DataFrame(
        {
            "rank": cudf.Series(
                np.repeat(np.arange(comm.nranks, dtype="int32"), rows_per_rank)
            ),
            "value": cudf.Series(values * comm.nranks, dtype="int32"),
        }
    )
    result_df = pylibcudf_to_cudf_dataframe(
        unpack_and_concat(gathered, br=br, stream=DEFAULT_STREAM),
        column_names=["rank", "value"],
    )
    assert_eq(result_df, expected_df)


# ---------------------------------------------------------------------------
# Shared pool fixture — module-scoped so the UCXX bootstrap runs once per
# nranks value and is reused across all tests that accept it.
# ---------------------------------------------------------------------------


@pytest.fixture(
    scope="module",
    params=[1, 2, 4],
    ids=["nranks=1", "nranks=2", "nranks=4"],
)
def mp_pool(request: pytest.FixtureRequest) -> Generator[MultiprocessingPool, None, None]:
    """Yield a live MultiprocessingPool for the parametrised worker count."""
    nranks: int = request.param
    with MultiprocessingPool(nranks=nranks, bind_verify=False) as pool:
        yield pool


# ---------------------------------------------------------------------------
# Tests using the shared pool fixture
# ---------------------------------------------------------------------------


def test_mp_pool_ucxx_cluster(mp_pool: MultiprocessingPool) -> None:
    """UCXX cluster is set up with the correct number of ranks on every worker."""
    nranks = mp_pool.nranks
    nranks_per_worker = mp_pool.run(_worker_comm_nranks)
    assert nranks_per_worker == [nranks] * nranks

    ranks = mp_pool.run(_worker_comm_rank)
    assert sorted(ranks) == list(range(nranks))


def test_mp_pool_worker_context(mp_pool: MultiprocessingPool) -> None:
    """WorkerContext is initialised and the communicator is valid on every worker."""
    nranks = mp_pool.nranks
    comm_strings = mp_pool.run(_worker_comm_str)
    assert len(comm_strings) == nranks
    for comm_str in comm_strings:
        assert f"nranks={nranks}" in comm_str


def test_mp_pool_lambda(mp_pool: MultiprocessingPool) -> None:
    """A lambda is broadcast to all workers via cloudpickle and executed."""
    results = mp_pool.run(lambda: 42)
    assert results == [42] * mp_pool.nranks


def test_mp_pool_closure(mp_pool: MultiprocessingPool) -> None:
    """A closure capturing a local variable is broadcast and executed correctly."""
    magic = 1337
    results = mp_pool.run(lambda: magic)
    assert results == [magic] * mp_pool.nranks


def test_mp_pool_error_propagation(mp_pool: MultiprocessingPool) -> None:
    """Exceptions raised on workers propagate as an ExceptionGroup, one per worker."""
    with pytest.raises(ExceptionGroup) as exc_info:
        mp_pool.run(_worker_raise)
    group = exc_info.value
    assert len(group.exceptions) == mp_pool.nranks
    for exc in group.exceptions:
        assert isinstance(exc, ValueError)


def test_mp_pool_error_propagation_partial(mp_pool: MultiprocessingPool) -> None:
    """ExceptionGroup contains only the exceptions from even-ranked workers."""
    n_even = math.ceil(mp_pool.nranks / 2)
    with pytest.raises(ExceptionGroup) as exc_info:
        mp_pool.run(_worker_raise_on_even_ranks)
    group = exc_info.value
    assert len(group.exceptions) == n_even
    for exc in group.exceptions:
        assert isinstance(exc, ValueError)
        assert "rank" in str(exc)


def test_mp_pool_multiple_tasks(mp_pool: MultiprocessingPool) -> None:
    """Multiple sequential run() calls complete without deadlock."""
    nranks = mp_pool.nranks
    for expected in [10, 20, 30]:
        results = mp_pool.run(lambda v=expected: v)
        assert results == [expected] * nranks


def test_mp_pool_resource_binding(mp_pool: MultiprocessingPool) -> None:
    """rrun.bind() is called and workers report a valid ResourceBinding."""
    bindings = mp_pool.run(_worker_check_binding)
    assert len(bindings) == mp_pool.nranks
    for b in bindings:
        assert isinstance(b, ResourceBinding)


@pytest.mark.parametrize("total_num_partitions", [1, 10])
def test_mp_pool_shuffle(mp_pool: MultiprocessingPool, total_num_partitions: int) -> None:
    """Distributed shuffle via the pool preserves all rows across all partitions."""
    num_rows = 100
    row_counts = mp_pool.run(_worker_run_shuffle, total_num_partitions, num_rows)
    assert sum(row_counts) == num_rows


def test_mp_pool_allgather_multirun(mp_pool: MultiprocessingPool) -> None:
    """Five sequential AllGather calls with unique per-run data; verified in-worker."""
    for run_id in range(5):
        mp_pool.run(_worker_allgather_and_verify, run_id, rows_per_rank=10)


# ---------------------------------------------------------------------------
# Tests that manage their own pool (lifecycle / configuration tests)
# ---------------------------------------------------------------------------


def test_mp_pool_oversubscription() -> None:
    """Pool starts correctly when nranks > ngpus (multiple workers per GPU)."""
    from cuda.core import system as cuda_system

    ngpus = cuda_system.get_num_devices()
    if ngpus == 0:
        pytest.skip("No CUDA devices available")
    nranks = ngpus * 2
    with MultiprocessingPool(nranks=nranks, bind_verify=False) as pool:
        nranks_per_worker = pool.run(_worker_comm_nranks)
        assert nranks_per_worker == [nranks] * nranks


def test_mp_pool_option_statistics() -> None:
    """Statistics option is forwarded and the worker context is valid."""
    options = Options({"mp_statistics": "true", "mp_print_statistics": "false"})
    with MultiprocessingPool(nranks=1, options=options, bind_verify=False) as pool:
        comm_strings = pool.run(_worker_comm_str)
        assert len(comm_strings) == 1
        assert "nranks=1" in comm_strings[0]


def test_mp_pool_shutdown_idempotent() -> None:
    """Calling shutdown() more than once does not raise."""
    pool = MultiprocessingPool(nranks=1, bind_verify=False)
    pool.shutdown()
    pool.shutdown()  # must not raise


def test_mp_pool_run_after_shutdown() -> None:
    """run() raises RuntimeError after the pool has been shut down."""
    pool = MultiprocessingPool(nranks=1, bind_verify=False)
    pool.shutdown()
    with pytest.raises(RuntimeError, match="shut down"):
        pool.run(lambda: None)


def test_mp_pool_repr() -> None:
    """repr() contains nranks and status information."""
    with MultiprocessingPool(nranks=2, bind_verify=False) as pool:
        r = repr(pool)
        assert "nranks=2" in r
        assert "alive" in r
    assert "shut down" in repr(pool)
