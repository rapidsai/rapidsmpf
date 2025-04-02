# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Shuffler integration for Dask Distributed clusters."""

from __future__ import annotations

_shuffle_counter: int = 0


def get_shuffle_id() -> int:
    """
    Return the unique id for a new shuffle.

    Returns
    -------
    The enumerated integer id for the current shuffle.
    """
    global _shuffle_counter  # noqa: PLW0603

    _shuffle_counter += 1
    return _shuffle_counter
