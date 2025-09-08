# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from ucxx._lib.libucxx import UCXAddress, UCXWorker

from rapidsmpf.communicator.communicator import Communicator
from rapidsmpf.config import Options

def new_communicator(
    nranks: int,
    ucx_worker: UCXWorker | None,
    root_ucxx_address: UCXAddress | None,
    options: Options,
) -> Communicator: ...
def get_root_ucxx_address(comm: Communicator) -> bytes: ...
def barrier(comm: Communicator) -> None: ...
