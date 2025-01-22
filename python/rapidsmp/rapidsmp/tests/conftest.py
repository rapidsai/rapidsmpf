# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from mpi4py import MPI

from rapidsmp.communicator.mpi import new_communicator

if TYPE_CHECKING:
    from collections.abc import Generator

    from rapidsmp.communicator.communicator import Communicator


@pytest.fixture(scope="session")
def _mpi_comm() -> Communicator:
    """
    Fixture for rapidsmp's MPI communicator to use throughout the session.

    This fixture provides a session-wide `Communicator` instance that wraps
    the MPI world communicator (`MPI.COMM_WORLD`).

    Do not use this fixture directly, use the `comm` fixture instead.
    """
    return new_communicator(MPI.COMM_WORLD)


@pytest.fixture
def comm(_mpi_comm: Communicator) -> Generator[Communicator, None, None]:
    """
    Fixture for a rapidsmp communicator, scoped for each test.
    """
    MPI.COMM_WORLD.barrier()
    yield _mpi_comm
    MPI.COMM_WORLD.barrier()
