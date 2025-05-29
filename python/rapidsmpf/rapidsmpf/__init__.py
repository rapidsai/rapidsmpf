# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Collection of multi-gpu, distributed memory algorithms."""

from __future__ import annotations

from rapidsmpf._version import __git_commit__, __version__  # noqa: F401

# If librapidsmpf was installed as a wheel, we must request it to load the
# library symbols. Otherwise, we assume that the library was installed in a
# system path that ld can find.
try:
    import librapidsmpf
except ModuleNotFoundError:
    pass
else:
    librapidsmpf.load_library()
    del librapidsmpf
