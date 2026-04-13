# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Submodule for communication abstraction (e.g. UCXX and MPI)."""

from __future__ import annotations

# Ensure ucxx Python module is imported so its __init__ runs and
# loads libucxx_python.so (required when installed via pip).
# The ucxx.pyx bindings only use `cimport`, so the Python module
# is otherwise never imported and its initialization is skipped.
import ucxx  # noqa: F401

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
