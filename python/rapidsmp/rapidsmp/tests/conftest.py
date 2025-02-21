# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from mpi4py import MPI

import rmm.mr

from rapidsmp.communicator.mpi import new_communicator
from rapidsmp.communicator.testing import ucxx_mpi_setup

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


@pytest.fixture(scope="session")
def _ucxx_comm() -> Communicator:
    """
    Fixture for rapidsmp's UCXX communicator to use throughout the session.

    This fixture provides a session-wide `Communicator` instance that wraps
    the an underlying UCXX communicator.

    Do not use this fixture directly, use the `ucxx_comm` fixture instead.
    """
    return ucxx_mpi_setup(None)


@pytest.fixture(
    params=["mpi", "ucxx"],
)
def comm(request) -> Generator[Communicator, None, None]:
    """
    Fixture for a rapidsmp communicator, scoped for each test.
    """
    MPI.COMM_WORLD.barrier()
    yield request.getfixturevalue(f"_{request.param}_comm")
    MPI.COMM_WORLD.barrier()


@pytest.fixture
def device_mr() -> Generator[rmm.mr.CudaMemoryResource, None, None]:
    """
    Fixture for creating a new cuda memory resource and making it the
    current rmm resource temporarily.
    """
    prior_mr = rmm.mr.get_current_device_resource()
    try:
        mr = rmm.mr.CudaMemoryResource()
        rmm.mr.set_current_device_resource(mr)
        yield mr
    finally:
        rmm.mr.set_current_device_resource(prior_mr)
