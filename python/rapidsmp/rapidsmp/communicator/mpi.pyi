# Copyright (c) 2025, NVIDIA CORPORATION.

from mpi4py.MPI import Intracomm

from rapidsmp.communicator.communicator import Communicator

def new_communicator(comm: Intracomm) -> Communicator: ...
