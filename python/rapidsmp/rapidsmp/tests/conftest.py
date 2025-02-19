# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import ucxx._lib.libucxx as ucx_api
from mpi4py import MPI

import rmm.mr

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


def _initialize_ucxx():
    """
    Initialize UCXX resources.
    """
    ucxx_context = ucx_api.UCXContext(
        feature_flags=(ucx_api.Feature.AM, ucx_api.Feature.TAG)
    )
    ucxx_worker = ucx_api.UCXWorker(ucxx_context)
    ucxx_worker.start_progress_thread(polling_mode=True)

    return ucxx_worker


def _ucxx_mpi_setup(ucxx_worker):
    """
    Bootstraps a UCXX communicator within an MPI rank.
    """
    from rapidsmp.communicator.ucxx import (
        barrier,
        get_root_ucxx_address,
        new_communicator,
    )

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

    barrier(comm)

    return comm


@pytest.fixture(scope="session")
def _ucxx_comm() -> Communicator:
    """
    Fixture for rapidsmp's UCXX communicator to use throughout the session.

    This fixture provides a session-wide `Communicator` instance that wraps
    the an underlying UCXX communicator.

    Do not use this fixture directly, use the `comm` fixture instead.
    """
    return _ucxx_mpi_setup(None)


@pytest.fixture
def ucxx_comm(_ucxx_comm: Communicator) -> Generator[Communicator, None, None]:
    """
    Fixture for a rapidsmp UCXX communicator, scoped for each test.
    """
    MPI.COMM_WORLD.barrier()
    yield _ucxx_comm
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
