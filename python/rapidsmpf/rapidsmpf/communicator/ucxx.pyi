# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from ucxx._lib.libucxx import UCXAddress, UCXWorker

from rapidsmpf.communicator.communicator import Communicator
from rapidsmpf.config import Options
from rapidsmpf.progress_thread import ProgressThread
from rapidsmpf.statistics import Statistics

def new_communicator(
    nranks: int,
    ucx_worker: UCXWorker | None,
    root_ucxx_address: UCXAddress | None,
    options: Options,
    progress: ProgressThread | Statistics | None = None,
) -> Communicator: ...
def get_root_ucxx_address(comm: Communicator) -> bytes: ...
def barrier(comm: Communicator) -> None: ...
