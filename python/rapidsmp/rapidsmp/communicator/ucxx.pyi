# Copyright (c) 2025, NVIDIA CORPORATION.

from ucxx._lib.libucxx import UCXAddress, UCXWorker

from rapidsmp.communicator.communicator import Communicator

def new_communicator(
    nranks: int, ucx_worker: UCXWorker, root_ucxx_address: UCXAddress
) -> Communicator: ...
def get_root_ucxx_address(comm: Communicator) -> str: ...
def barrier(comm: Communicator) -> None: ...
