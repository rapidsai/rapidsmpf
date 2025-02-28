# Copyright (c) 2025, NVIDIA CORPORATION.
from __future__ import annotations

import pytest
from mpi4py import MPI

from rapidsmp.communicator.communicator import LOG_LEVEL
from rapidsmp.communicator.mpi import new_communicator
from rapidsmp.communicator.testing import initialize_ucxx, ucxx_mpi_setup


@pytest.mark.parametrize("level", LOG_LEVEL)
def test_log_level(capfd, level: LOG_LEVEL):
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("RAPIDSMP_LOG", level.name)

        comm = new_communicator(MPI.COMM_WORLD)
        assert comm.logger.verbosity_level is level
        comm.logger.print("PRINT")
        comm.logger.warn("WARN")
        comm.logger.info("INFO")
        comm.logger.debug("DEBUG")
        comm.logger.trace("TRACE")
        output = capfd.readouterr().out
        # Iff the verbosity level is high enough for a log category, we expect the
        # category name to be printed exactly twice. E.g., a warning would look like:
        # "[WARN:0:0] WARN".
        assert output.count("NONE") == 0
        assert output.count("PRINT") == (2 if level >= int(LOG_LEVEL.PRINT) else 0)
        assert output.count("WARN") == (2 if level >= int(LOG_LEVEL.WARN) else 0)
        assert output.count("INFO") == (2 if level >= int(LOG_LEVEL.INFO) else 0)
        assert output.count("DEBUG") == (2 if level >= int(LOG_LEVEL.DEBUG) else 0)
        assert output.count("TRACE") == (2 if level >= int(LOG_LEVEL.TRACE) else 0)


def test_mpi():
    comm = new_communicator(MPI.COMM_WORLD)
    assert comm.nranks == MPI.COMM_WORLD.size
    assert comm.rank == MPI.COMM_WORLD.rank


def test_ucxx():
    ucxx_worker = initialize_ucxx()
    comm = ucxx_mpi_setup(ucxx_worker)
    assert comm.nranks == MPI.COMM_WORLD.size

    ucxx_worker.stop_progress_thread()
