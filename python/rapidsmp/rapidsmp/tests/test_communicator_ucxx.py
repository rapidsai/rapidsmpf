# Copyright (c) 2025, NVIDIA CORPORATION.
from __future__ import annotations

import pytest

# import ucxx
import ucxx._lib.libucxx as ucx_api
from mpi4py import MPI

from rapidsmp.communicator.ucxx import (
    barrier,
    get_root_ucxx_address,
    new_communicator,
)


def ucxx_mpi_setup(ucxx_worker):
    if MPI.COMM_WORLD.Get_rank() == 0:
        comm = new_communicator(MPI.COMM_WORLD.size, ucxx_worker)
        root_address_str = get_root_ucxx_address(comm)
    else:
        root_address_str = None

    root_address_str = MPI.COMM_WORLD.bcast(root_address_str, root=0)

    if MPI.COMM_WORLD.Get_rank() != 0:
        root_address = ucx_api.UCXAddress.create_from_buffer(root_address_str)
        comm = new_communicator(MPI.COMM_WORLD.size, ucxx_worker, root_address)

    assert comm.nranks == MPI.COMM_WORLD.size

    return comm


def initialize_ucxx():
    # ucxx_worker = ucxx.core._get_ctx().worker
    ucxx_context = ucx_api.UCXContext(
        feature_flags=(ucx_api.Feature.AM, ucx_api.Feature.TAG)
    )
    ucxx_worker = ucx_api.UCXWorker(ucxx_context)
    ucxx_worker.start_progress_thread(polling_mode=True)

    return ucxx_worker


@pytest.mark.parametrize("RAPIDSMP_LOG", range(5))
def test_mpi(capfd, RAPIDSMP_LOG):
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("RAPIDSMP_LOG", str(RAPIDSMP_LOG))

        ucxx_worker = initialize_ucxx()
        comm = ucxx_mpi_setup(ucxx_worker)

        # barrier(comm)

        assert comm.nranks == MPI.COMM_WORLD.size
        # Ranks assigned by UCXX do not necessarily match MPI's
        # assert comm.rank == MPI.COMM_WORLD.rank
        assert comm.logger.verbosity_level == RAPIDSMP_LOG

        comm.logger.warn("WARN")
        comm.logger.info("INFO")
        comm.logger.debug("DEBUG")
        comm.logger.trace("TRACE")
        output = capfd.readouterr().out

        # Iff the verbosity level is high enough for a log category, we expect the
        # category name to be printed exactly twice. E.g., a warning would look like:
        # "[WARN:0:0] WARN".
        assert output.count("WARN") == (2 if RAPIDSMP_LOG > 0 else 0)
        assert output.count("INFO") == (2 if RAPIDSMP_LOG > 1 else 0)
        assert output.count("DEBUG") == (2 if RAPIDSMP_LOG > 2 else 0)
        assert output.count("TRACE") == (2 if RAPIDSMP_LOG > 3 else 0)

        barrier(comm)

        ucxx_worker.stop_progress_thread()
