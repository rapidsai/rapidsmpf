# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
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


__all__ = [
    "COMMUNICATORS",
]
