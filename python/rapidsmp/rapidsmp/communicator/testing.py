# Copyright (c) 2025, NVIDIA CORPORATION.
"""Submodule for testing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import ucxx._lib.libucxx as ucx_api
from mpi4py import MPI

if TYPE_CHECKING:
    from rapidsmp.communicator.communicator import Communicator


def initialize_ucxx() -> ucx_api.UCXWorker:
    """
    Initialize UCXX resources.

    Returns
    -------
    ucx_api.UCXWorker
        A new UCXWorker.
    """
    ucxx_context = ucx_api.UCXContext(
        feature_flags=(ucx_api.Feature.AM, ucx_api.Feature.TAG)
    )
    ucxx_worker = ucx_api.UCXWorker(ucxx_context)
    ucxx_worker.start_progress_thread(polling_mode=True)

    return ucxx_worker


def ucxx_mpi_setup(ucxx_worker: ucx_api.UCXWorker) -> Communicator:
    """
    Bootstrap a UCXX communicator within an MPI rank.

    Parameters
    ----------
    ucxx_worker
        An existing UCXX worker to use.

    Returns
    -------
    Communicator
        A new rapidsmp-ucxx communicator.
    """
    from rapidsmp.communicator.ucxx import (
        barrier,
        get_root_ucxx_address,
        new_communicator,
    )

    if MPI.COMM_WORLD.Get_rank() == 0:
        comm = new_communicator(MPI.COMM_WORLD.size, ucxx_worker, None)
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
