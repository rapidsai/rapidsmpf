# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from rapidsmpf.communicator.communicator import Communicator
from rapidsmpf.statistics import Statistics

def gather_statistics(
    comm: Communicator,
    op_id: int,
    stats: Statistics,
    root: int = 0,
) -> list[Statistics]: ...
