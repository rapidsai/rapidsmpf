# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import numpy as np
import pytest

from rapidsmpf.shuffler import PartitionAssignment
from rapidsmpf.streaming.chunks.partition import (
    PartitionMapChunk,
    PartitionVectorChunk,
)
from rapidsmpf.streaming.coll.shuffler import ShufflerAsync
from rapidsmpf.streaming.core.actor import define_actor, run_actor_network
from rapidsmpf.streaming.core.leaf_actor import pull_from_channel
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.testing import (
    generate_packed_data,
    make_partition_data,
    validate_packed_data,
    validate_partition_data,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from rmm.pylibrmm.stream import Stream

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.actor import CppActor
    from rapidsmpf.streaming.core.channel import Channel
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


@define_actor()
async def generate_inputs(
    context: Context,
    ch: Channel[PartitionMapChunk],
    num_rows: int,
    num_chunks: int,
    num_partitions: int,
) -> None:
    br = context.br()
    for i in range(num_chunks):
        stream = context.get_stream_from_pool()
        data = {
            pid: generate_packed_data(
                num_rows, (i * num_partitions + pid) * num_rows, stream, br
            )
            for pid in range(num_partitions)
        }
        msg = Message(i, PartitionMapChunk.from_packed_data_map(data, br))
        await ch.send(context, msg)
    await ch.drain(context)


@define_actor()
async def do_shuffle(
    context: Context,
    comm: Communicator,
    ch_in: Channel[PartitionMapChunk],
    ch_out: Channel[PartitionVectorChunk],
    op_id: int,
    num_partitions: int,
    *,
    partition_assignment: PartitionAssignment = PartitionAssignment.ROUND_ROBIN,
) -> None:
    shuffle = ShufflerAsync(
        context, comm, op_id, num_partitions, partition_assignment=partition_assignment
    )
    while (msg := await ch_in.recv(context)) is not None:
        chunk = PartitionMapChunk.from_message(msg, br=context.br())
        shuffle.insert(chunk.to_packed_data_map())
    await shuffle.insert_finished(context)
    for pid in shuffle.local_partitions():
        data = shuffle.extract(pid)
        out_chunk = PartitionVectorChunk.from_packed_data_list(data, context.br())
        await ch_out.send(context, Message(pid, out_chunk))
    await ch_out.drain(context)


@pytest.mark.parametrize("num_partitions", [4, 8])
def test_shuffler_runtime_obeys_contiguous_assignment(
    context: Context,
    comm: Communicator,
    num_partitions: int,
) -> None:
    if comm.nranks != 1:
        pytest.skip("Only support single-rank runs")

    actors: list[CppActor | Awaitable[None]] = []

    num_rows = 200
    num_chunks = 3
    op_id = 0
    ch_in: Channel[PartitionMapChunk] = context.create_channel()
    actors.append(generate_inputs(context, ch_in, num_rows, num_chunks, num_partitions))
    ch_shuffled: Channel[PartitionVectorChunk] = context.create_channel()
    actors.append(
        do_shuffle(
            context,
            comm,
            ch_in,
            ch_shuffled,
            op_id,
            num_partitions,
            partition_assignment=PartitionAssignment.CONTIGUOUS,
        )
    )
    actor, deferred = pull_from_channel(context, ch_shuffled)
    actors.append(actor)

    run_actor_network(context, actors=actors)
    messages = deferred.release()
    received_pids = [msg.sequence_number for msg in messages]

    # Single rank, so every partition is local to this rank.
    assert set(received_pids) == set(range(num_partitions))

    # Validate the data routed to each local partition. Across the ``num_chunks``
    # inputs, partition ``pid`` receives the packed sequence ``generate_inputs``
    # produced for ``(chunk i, pid)``, which starts at value
    # ``(i * num_partitions + pid) * num_rows``. The shuffler makes no ordering
    # guarantee, so match each received chunk to its expected input by start value.
    for msg in messages:
        pid = msg.sequence_number
        packed = PartitionVectorChunk.from_message(
            msg, br=context.br()
        ).to_packed_data_list()
        assert len(packed) == num_chunks
        by_offset = {
            int(np.frombuffer(pd.to_host_bytes(), dtype=np.int64)[0]): pd
            for pd in packed
        }
        for i in range(num_chunks):
            offset = (i * num_partitions + pid) * num_rows
            validate_packed_data(by_offset[offset], num_rows, offset)


def test_shuffler_object_interface(
    context: Context,
    comm: Communicator,
) -> None:
    if comm.nranks != 1:
        pytest.skip("Only support single-rank runs")
    actors: list[CppActor | Awaitable[None]] = []

    num_partitions = 5
    num_rows = 100
    num_chunks = 4
    op_id = 0
    ch_in: Channel[PartitionMapChunk] = context.create_channel()
    actors.append(generate_inputs(context, ch_in, num_rows, num_chunks, num_partitions))
    ch_shuffled: Channel[PartitionVectorChunk] = context.create_channel()
    actors.append(
        do_shuffle(
            context,
            comm,
            ch_in,
            ch_shuffled,
            op_id,
            num_partitions,
        )
    )
    actor, deferred = pull_from_channel(context, ch_shuffled)
    actors.append(actor)

    run_actor_network(context, actors=actors)
    messages = deferred.release()
    # TODO: single rank only assertions
    assert len(messages) == num_partitions
    assert [msg.sequence_number for msg in messages] == list(range(num_partitions))
    chunks = [
        (msg.sequence_number, PartitionVectorChunk.from_message(msg, br=context.br()))
        for msg in messages
    ]

    # Each destination partition ``pid`` receives, across the ``num_chunks`` inputs,
    # the packed sequence generated by ``generate_inputs`` for ``(chunk i, pid)``,
    # which starts at value ``(i * num_partitions + pid) * num_rows``. The shuffler
    # makes no ordering guarantee, so match each received chunk to its expected
    # input chunk by starting value.
    for pid, vec_chunk in chunks:
        packed = vec_chunk.to_packed_data_list()
        assert len(packed) == num_chunks
        by_offset = {
            int(np.frombuffer(pd.to_host_bytes(), dtype=np.int64)[0]): pd
            for pd in packed
        }
        for i in range(num_chunks):
            offset = (i * num_partitions + pid) * num_rows
            validate_packed_data(by_offset[offset], num_rows, offset)
