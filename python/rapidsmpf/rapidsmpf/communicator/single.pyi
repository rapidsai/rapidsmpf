# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from rapidsmpf.communicator.communicator import Communicator
from rapidsmpf.config import Options
from rapidsmpf.progress_thread import ProgressThread
from rapidsmpf.statistics import Statistics

def new_communicator(
    options: Options,
    progress: ProgressThread | Statistics | None = None,
) -> Communicator: ...
