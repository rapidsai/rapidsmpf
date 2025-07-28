# Copyright (c) 2025, NVIDIA CORPORATION.

from enum import IntEnum
from typing import cast

from rapidsmp.communicator.communicator import Communicator
from ucxx._lib.libucxx import UCXAddress, UCXWorker

class ProgressMode(IntEnum):
    Blocking = cast(int, ...)
    Polling = cast(int, ...)
    ThreadBlocking = cast(int, ...)
    ThreadPolling = cast(int, ...)

def new_communicator(
    nranks: int,
    ucx_worker: UCXWorker,
    root_ucxx_address: UCXAddress,
    progress_mode: ProgressMode = ...,
) -> Communicator: ...
def get_root_ucxx_address(comm: Communicator) -> str: ...
def barrier(comm: Communicator) -> None: ...
