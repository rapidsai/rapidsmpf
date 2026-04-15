# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Tests for gather_statistics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rapidsmpf.coll import gather_statistics
from rapidsmpf.statistics import Statistics

if TYPE_CHECKING:
    from rapidsmpf.communicator.communicator import Communicator


def test_gather_basic(comm: Communicator) -> None:
    stats = Statistics(enable=True)
    stats.add_stat("x", float(comm.rank))

    others = gather_statistics(comm, 0, stats)

    if comm.rank == 0:
        assert len(others) == comm.nranks - 1
        for s in others:
            assert s.enabled
            assert len(s.list_stat_names()) == 1
    else:
        assert len(others) == 0


def test_gather_single_rank(comm: Communicator) -> None:
    if comm.nranks != 1:
        return

    stats = Statistics(enable=True)
    stats.add_stat("x", 42.0)

    others = gather_statistics(comm, 1, stats)
    assert len(others) == 0


def test_gather_disjoint_names(comm: Communicator) -> None:
    stats = Statistics(enable=True)
    stats.add_stat(f"rank-{comm.rank}", 1.0)

    others = gather_statistics(comm, 2, stats)

    if comm.rank == 0:
        assert len(others) == comm.nranks - 1
        # Each remote rank contributed a uniquely named stat.
        all_names: set[str] = set(stats.list_stat_names())
        for s in others:
            all_names.update(s.list_stat_names())
        assert len(all_names) == comm.nranks
        for r in range(comm.nranks):
            assert f"rank-{r}" in all_names
    else:
        assert len(others) == 0
