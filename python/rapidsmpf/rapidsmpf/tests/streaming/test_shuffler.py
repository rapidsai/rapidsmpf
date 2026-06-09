# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from rapidsmpf.streaming.coll.shuffler import ShufflerAsync
from rapidsmpf.testing import (
    generate_packed_data,
    make_partition_data,
    validate_partition_data,
)

if TYPE_CHECKING:
    from rmm.pylibrmm.stream import Stream

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.context import Context


@pytest.mark.parametrize("total_num_partitions", [1, 2, 5, 10])
@pytest.mark.parametrize("total_num_rows", [1, 100, 1000])
def test_shuffler_round_trip(
    context: Context,
    comm: Communicator,
    stream: Stream,
    total_num_partitions: int,
    total_num_rows: int,
) -> None:
    """
    End-to-end correctness of the async streaming shuffler.

    Each rank inserts the input regions it owns and, after shuffling, every local
    partition is validated against the conserved data model.
    """
    br = context.br()
    shuffler = ShufflerAsync(context, comm, 0, total_num_partitions)

    for local_pidx in shuffler.local_partitions():
        chunks = make_partition_data(
            total_num_partitions, total_num_rows, local_pidx, stream, br
        )
        if chunks:
            shuffler.insert(chunks)

    asyncio.run(shuffler.insert_finished(context))

    for local_pidx in shuffler.local_partitions():
        validate_partition_data(
            shuffler.extract(local_pidx),
            total_num_partitions,
            total_num_rows,
            local_pidx,
        )


@pytest.mark.parametrize("n_inserts", [1, 10])
@pytest.mark.parametrize("n_partitions", [1, 10, 100])
def test_shuffler_insert_wait_extract(
    context: Context,
    comm: Communicator,
    stream: Stream,
    n_inserts: int,
    n_partitions: int,
) -> None:
    """
    Each rank inserts ``n_inserts`` full partition maps; after shuffling each local
    partition must receive exactly ``n_inserts * nranks`` chunks.
    """
    n_elements = 100
    br = context.br()
    shuffler = ShufflerAsync(context, comm, 0, n_partitions)

    for _ in range(n_inserts):
        data = {
            pid: generate_packed_data(n_elements, 0, stream, br)
            for pid in range(n_partitions)
        }
        shuffler.insert(data)

    asyncio.run(shuffler.insert_finished(context))

    local_pids = shuffler.local_partitions()

    finished_pids = []
    n_chunks_received = 0
    for pid in local_pids:
        chunks = shuffler.extract(pid)
        n_chunks_received += len(chunks)
        finished_pids.append(pid)

    assert n_chunks_received == n_inserts * len(local_pids) * comm.nranks
    assert finished_pids == local_pids
