# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from mpi4py.MPI import Intracomm

from rapidsmpf.communicator.communicator import Communicator
from rapidsmpf.config import Options

def new_communicator(comm: Intracomm, options: Options) -> Communicator: ...
