# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Submodule for communication abstraction (e.g. UCXX and MPI)."""

from __future__ import annotations

from rapidsmpf.communicator.communicator import _available_communicators

COMMUNICATORS = _available_communicators()
"""Tuple of available communicators.

RapidsMPF includes a collection of communicator backends, available as submodules
under ``rapidsmpf.communicator.*``. Typically, the Conda distribution includes
both UCXX and MPI support, while the PIP installation generally supports only UCXX.
"""
try:
    # Ensure that we don't initialise MPI when importing types from
    # mpi4py.MPI
    import mpi4py

    mpi4py.rc.initialize = False
except ImportError:
    pass


__all__ = [
    "COMMUNICATORS",
]
