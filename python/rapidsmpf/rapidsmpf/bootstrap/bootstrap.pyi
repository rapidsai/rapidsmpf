# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum

from rapidsmpf.communicator.communicator import Communicator
from rapidsmpf.config import Options

class Backend(IntEnum):
    AUTO = ...
    FILE = ...

def create_ucxx_comm(
    backend: Backend = ...,
    options: Options | None = ...,
) -> Communicator: ...
def is_running_with_rrun() -> bool: ...
def get_nranks() -> int: ...
