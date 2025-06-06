# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from rapidsmpf.communicator.communicator import Communicator
from rapidsmpf.statistics import Statistics

class ProgressThread:
    def __init__(
        self,
        comm: Communicator,
        statistics: Statistics | None = None,
    ) -> None: ...
