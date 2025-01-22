# Copyright (c) 2025, NVIDIA CORPORATION.
from __future__ import annotations

from mpi4py import MPI

from rapidsmp.communicator.mpi import new_communicator


def test_mpi():
    comm = new_communicator(MPI.COMM_WORLD)
    assert comm.nranks == MPI.COMM_WORLD.size
    assert comm.rank == MPI.COMM_WORLD.rank
