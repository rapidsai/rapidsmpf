# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Submodule for communication abstraction (e.g. UCXX and MPI)."""

from __future__ import annotations

from rapidsmpf.communicator.communicator import _available_communicators

COMMUNICATORS = _available_communicators()
"""Tuple of available communicators.

RapidsMPF if built with a set of communicators, which can be found as submodules
under ``rapidsmpf.communicator.*``. Typically, the Conda package is built with
both UCXX and MPI support whereas the PIP package only supports UCXX.
"""


__all__ = [
    "COMMUNICATORS",
]
