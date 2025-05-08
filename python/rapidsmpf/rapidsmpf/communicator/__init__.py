# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Submodule for communication abstraction (e.g. UCXX and MPI)."""

from __future__ import annotations

import importlib.util

MPI_SUPPORT: bool = importlib.util.find_spec("rapidsmpf.communicator.mpi") is not None
"""Whether the MPI communicator (``rapidsmpf.communicator.mpi``) is available.

This is false when RapidsMPF wasn't built with MPI support. Typically, MPI is
supported in the Conda package but not in the PIP package.
"""


__all__ = [
    "MPI_SUPPORT",
]
