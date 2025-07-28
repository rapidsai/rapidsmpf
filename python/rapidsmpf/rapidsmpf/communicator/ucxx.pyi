# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum
from typing import cast

from ucxx._lib.libucxx import UCXAddress, UCXWorker

from rapidsmpf.communicator.communicator import Communicator
from rapidsmpf.config import Options

class ProgressMode(IntEnum):
    Blocking = cast(int, ...)
    Polling = cast(int, ...)
    ThreadBlocking = cast(int, ...)
    ThreadPolling = cast(int, ...)

def new_communicator(
    nranks: int,
    ucx_worker: UCXWorker,
    root_ucxx_address: UCXAddress,
    options: Options,
    progress_mode: ProgressMode = ...,
) -> Communicator: ...
def get_root_ucxx_address(comm: Communicator) -> bytes: ...
def barrier(comm: Communicator) -> None: ...
