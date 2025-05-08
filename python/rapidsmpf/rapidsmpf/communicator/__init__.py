# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Submodule for communication abstraction (e.g. UCXX and MPI).

Attributes
----------
MPI_SUPPORT
    Whether the MPI communicator (`rapidsmpf.communicator.mpi`) is available. This is
    False when RapidsMPF wasn't built with MPI support. Typically, MPI is supported in
    the Conda package but not in the PIP package.
"""

from __future__ import annotations

import importlib.util

MPI_SUPPORT: bool = importlib.util.find_spec("rapidsmpf.communicator.mpi") is not None

__all__ = [
    "MPI_SUPPORT",
]
