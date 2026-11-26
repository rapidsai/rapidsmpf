# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for `Communicator.abort()`."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import textwrap
from typing import TYPE_CHECKING

import pytest

from rapidsmpf.coll import AllGather
from rapidsmpf.communicator import COMMUNICATORS
from rapidsmpf.communicator.single import new_communicator as single_communicator
from rapidsmpf.config import Options
from rapidsmpf.error import CommunicatorAborted
from rapidsmpf.memory.buffer_resource import BufferResource
from rapidsmpf.progress_thread import ProgressThread

if TYPE_CHECKING:
    from pathlib import Path

    import rmm.mr

# Generous upper bound, aborting is expected to take well under a second.
TIMEOUT = 60

# MPI communicators that must not be freed, see `test_mpi_abort_propagates_to_all_ranks`.
_never_freed: list[object] = []


def test_single_abort(capfd: pytest.CaptureFixture[str]) -> None:
    comm = single_communicator(Options(), ProgressThread())
    assert not comm.aborted
    comm.abort("injected failure")
    assert comm.aborted
    assert "RapidsMPF rank 0 aborting communicator: injected failure" in (
        capfd.readouterr().err
    )
    comm.abort("aborting again is a no-op")
    assert comm.aborted


@pytest.mark.skipif("mpi" not in COMMUNICATORS, reason="MPI not available")
def test_mpi_abort_propagates_to_all_ranks(
    device_mr: rmm.mr.CudaMemoryResource,
) -> None:
    MPI = pytest.importorskip("mpi4py.MPI")
    if MPI.COMM_WORLD.size < 2:
        pytest.skip("Requires at least 2 ranks")
    from rapidsmpf.communicator.mpi import new_communicator

    # Abort a dedicated communicator, not the one shared by the other tests.
    mpi_comm = MPI.COMM_WORLD.Dup()
    comm = new_communicator(mpi_comm, Options(), ProgressThread())
    allgather = AllGather(comm, 0, BufferResource(device_mr))
    # Rank 0 never finishes the allgather, so it only completes by aborting.
    if comm.rank == 0:
        comm.abort("injected failure")
    else:
        allgather.insert_finished()
    try:
        with pytest.raises(
            CommunicatorAborted,
            match="communicator aborted by rank 0: injected failure",
        ):
            allgather.wait_and_extract(timeout_ms=TIMEOUT * 1000)
    finally:
        if not comm.aborted:
            # Not notified: abort locally, so that the allgather fails and can be
            # destroyed instead of hanging.
            comm.abort("abort not propagated")
            with contextlib.suppress(CommunicatorAborted):
                allgather.wait_and_extract(timeout_ms=TIMEOUT * 1000)
    assert comm.aborted
    # `mpi_comm` is intentionally never freed (and kept alive): it may still hold
    # unmatched messages, and Open MPI fails when the context of a freed communicator
    # with unmatched messages is reused.
    _never_freed.append(mpi_comm)


# The remaining tests run each rank in a child process, independent of any MPI
# launcher, either because the processes are terminated or to bootstrap UCXX.


def _child_env() -> dict[str, str]:
    """Environment for child processes, hiding any MPI/PMIx launcher."""
    return {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("OMPI_", "PMIX_", "PMI_", "MPI_", "SLURM_", "RRUN_"))
    }


# Prepended to every child script, so that terminating does not leave core dumps.
_DISABLE_CORE_DUMPS = """
import resource
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
"""


def _spawn(script: str, *args: str) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-c", _DISABLE_CORE_DUMPS + textwrap.dedent(script), *args],
        env=_child_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _wait(proc: subprocess.Popen[str]) -> tuple[int, str, str]:
    try:
        stdout, stderr = proc.communicate(timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        pytest.fail(f"process did not exit, stdout:\n{stdout}\nstderr:\n{stderr}")
    return proc.returncode, stdout, stderr


def test_single_abort_terminate() -> None:
    proc = _spawn(
        """
        from rapidsmpf.communicator.single import new_communicator
        from rapidsmpf.config import Options
        from rapidsmpf.progress_thread import ProgressThread

        comm = new_communicator(Options(), ProgressThread())
        comm.abort("injected failure", terminate=True)
        raise SystemExit("abort() returned")
        """
    )
    returncode, _, stderr = _wait(proc)
    assert returncode == -signal.SIGABRT, stderr
    assert "RapidsMPF rank 0 terminating all ranks: injected failure" in stderr


# Bootstraps a 2-rank UCXX communicator, the root address is passed via a file. Then,
# rank 1 aborts while both ranks are in an allgather that rank 0 never finishes.
_UCXX_SCRIPT = """
import sys
import time
from pathlib import Path

import rmm.mr
import ucxx._lib.libucxx as ucx_api

from rapidsmpf.coll import AllGather
from rapidsmpf.communicator.ucxx import (
    barrier,
    get_root_ucxx_address,
    new_communicator,
)
from rapidsmpf.config import Options
from rapidsmpf.error import CommunicatorAborted
from rapidsmpf.memory.buffer_resource import BufferResource
from rapidsmpf.progress_thread import ProgressThread

rank = int(sys.argv[1])
root_address_file = Path(sys.argv[2])
terminate = sys.argv[3] == "terminate"
if rank == 0:
    comm = new_communicator(2, None, None, Options(), ProgressThread())
    tmp_file = root_address_file.with_suffix(".tmp")
    tmp_file.write_bytes(get_root_ucxx_address(comm))
    tmp_file.rename(root_address_file)
else:
    while not root_address_file.exists():
        time.sleep(0.01)
    root_address = ucx_api.UCXAddress.create_from_buffer(root_address_file.read_bytes())
    comm = new_communicator(2, None, root_address, Options(), ProgressThread())
barrier(comm)

allgather = AllGather(comm, 0, BufferResource(rmm.mr.CudaMemoryResource()))
if rank == 1:
    allgather.insert_finished()
    comm.abort("injected failure", terminate=terminate)
try:
    allgather.wait_and_extract(timeout_ms=60_000)
except CommunicatorAborted as e:
    print(f"rank {rank} caught: {e}")
finally:
    print(f"rank {rank} unwound")
"""


@pytest.mark.skipif("ucxx" not in COMMUNICATORS, reason="UCXX not available")
@pytest.mark.parametrize("terminate", [False, True], ids=["cooperative", "terminate"])
def test_ucxx_abort_propagates_to_peers(tmp_path: Path, *, terminate: bool) -> None:
    mode = "terminate" if terminate else "cooperative"
    procs = [
        _spawn(_UCXX_SCRIPT, str(rank), str(tmp_path / "root_address"), mode)
        for rank in range(2)
    ]
    try:
        results = [_wait(proc) for proc in procs]
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
                proc.communicate()
    for rank, (returncode, stdout, stderr) in enumerate(results):
        if terminate:
            # Both ranks are terminated without unwinding.
            assert returncode == -signal.SIGABRT, stderr
            assert "unwound" not in stdout
        else:
            # Both ranks raise and unwind normally.
            assert returncode == 0, stderr
            assert (
                f"rank {rank} caught: communicator aborted by rank 1: injected failure"
                in stdout
            ), stdout
            assert f"rank {rank} unwound" in stdout
    if terminate:
        assert (
            "RapidsMPF rank 1 terminating all ranks: injected failure" in results[1][2]
        )
        assert (
            "RapidsMPF rank 0 terminating: rank 1 aborted: injected failure"
            in results[0][2]
        )
