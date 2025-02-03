# Copyright (c) 2025, NVIDIA CORPORATION.
from __future__ import annotations

import pytest
from mpi4py import MPI

from rapidsmp.communicator.mpi import new_communicator


@pytest.mark.parametrize("RAPIDSMP_LOG", range(5))
def test_mpi(capfd, RAPIDSMP_LOG):
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("RAPIDSMP_LOG", str(RAPIDSMP_LOG))

        comm = new_communicator(MPI.COMM_WORLD)
        assert comm.nranks == MPI.COMM_WORLD.size
        assert comm.rank == MPI.COMM_WORLD.rank
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
