# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from rapidsmpf.progress_thread import ProgressThread

MPI = pytest.importorskip("mpi4py.MPI")
from typing import TYPE_CHECKING  # noqa: E402

from rapidsmpf.communicator.communicator import LOG_LEVEL  # noqa: E402
from rapidsmpf.communicator.mpi import new_communicator  # noqa: E402
from rapidsmpf.communicator.single import (  # noqa: E402
    new_communicator as single_process_comm,
)
from rapidsmpf.communicator.testing import initialize_ucxx, ucxx_mpi_setup  # noqa: E402
from rapidsmpf.config import Options, get_environment_variables  # noqa: E402

if TYPE_CHECKING:
    from rapidsmpf.communicator.communicator import Logger


@pytest.mark.parametrize(
    "level", list(LOG_LEVEL), ids=[level.name for level in LOG_LEVEL]
)
def test_log_level(capfd: pytest.CaptureFixture[str], level: LOG_LEVEL) -> None:
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("RAPIDSMPF_LOG", level.name)

        comm = new_communicator(
            MPI.COMM_WORLD, Options(get_environment_variables()), ProgressThread()
        )
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
        assert output.count("PRINT") == (2 if level >= LOG_LEVEL.PRINT else 0)
        assert output.count("WARN") == (2 if level >= LOG_LEVEL.WARN else 0)
        assert output.count("INFO") == (2 if level >= LOG_LEVEL.INFO else 0)
        assert output.count("DEBUG") == (2 if level >= LOG_LEVEL.DEBUG else 0)
        assert output.count("TRACE") == (2 if level >= LOG_LEVEL.TRACE else 0)


def test_mpi() -> None:
    comm = new_communicator(
        MPI.COMM_WORLD, Options(get_environment_variables()), ProgressThread()
    )
    assert comm.nranks == MPI.COMM_WORLD.size
    assert comm.rank == MPI.COMM_WORLD.rank


def test_ucxx() -> None:
    ucxx_worker = initialize_ucxx()
    comm = ucxx_mpi_setup(
        ucxx_worker, Options(get_environment_variables()), ProgressThread()
    )
    assert comm.nranks == MPI.COMM_WORLD.size

    ucxx_worker.stop_progress_thread()


def test_single_process() -> None:
    comm = single_process_comm(Options(get_environment_variables()), ProgressThread())
    assert comm.nranks == 1
    assert comm.rank == 0


def test_logger_weakref_raises() -> None:
    def get_logger() -> Logger:
        comm = single_process_comm(Options(), ProgressThread())
        return comm.logger

    with pytest.raises(RuntimeError):
        get_logger().print("Reference should be dead")
