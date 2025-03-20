# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from mpi4py.MPI import Intracomm

from rapidsmp.communicator.communicator import Communicator

def new_communicator(comm: Intracomm) -> Communicator: ...
