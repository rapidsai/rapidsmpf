# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Basic streaming example."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import cudf
import rmm.mr
from rmm.pylibrmm.stream import DEFAULT_STREAM

from rapidsmpf.buffer.resource import BufferResource
from rapidsmpf.communicator.single import (
    new_communicator as single_process_comm,
)
from rapidsmpf.config import Options, get_environment_variables
from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor
from rapidsmpf.streaming.core.channel import Channel, Message
from rapidsmpf.streaming.core.context import Context
from rapidsmpf.streaming.core.leaf_node import pull_from_channel, push_to_channel
from rapidsmpf.streaming.core.node import (
    define_py_node,
    run_streaming_pipeline,
)
from rapidsmpf.streaming.cudf.table_chunk import TableChunk
from rapidsmpf.utils.cudf import cudf_to_pylibcudf_table

if TYPE_CHECKING:
    from rapidsmpf.streaming.core.node import CppNode, PyNode


def main() -> int:
    """Basic example of a streaming pipeline."""
    # Initialize configuration options from environment variables.
    options = Options(get_environment_variables())

    # Create a context that will be used by all streaming nodes.
    ctx = Context(
        comm=single_process_comm(options),
        br=BufferResource(RmmResourceAdaptor(rmm.mr.get_current_device_resource())),
        options=options,
    )

    # Executor for Python nodes (asyncio coroutines).
    py_executor = ThreadPoolExecutor(max_workers=1)

    # Create some pylibcudf tables as input to the streaming pipeline.
    tables = [
        cudf_to_pylibcudf_table(cudf.DataFrame({"a": [1 * seq, 2 * seq, 3 * seq]}))
        for seq in range(10)
    ]

    # Wrap tables in TableChunk objects before sending them into the pipeline.
    # A TableChunk contains a pylibcudf table, a sequence number, and a CUDA stream.
    table_chunks = [
        Message(TableChunk.from_pylibcudf_table(seq, expect, DEFAULT_STREAM))
        for seq, expect in enumerate(tables)
    ]

    # Create input and output channels for table chunks.
    ch1: Channel[TableChunk] = Channel()
    ch2: Channel[TableChunk] = Channel()

    # Node 1: producer that pushes messages into the pipeline.
    # This is a native C++ node that runs as a coroutine with minimal Python overhead.
    node1: CppNode = push_to_channel(ctx, ch_out=ch1, messages=table_chunks)

    # Node 2: Python node that counts the total number of rows.
    # Runs as a Python coroutine (asyncio), which comes with overhead,
    # but releases the GIL on `await` and when calling into C++ APIs.
    @define_py_node()
    async def count_num_rows(
        ctx: Context, ch_in: Channel, ch_out: Channel, total_num_rows: list[int]
    ) -> None:
        assert len(total_num_rows) == 1, "should be a scalar"
        msg: Message[TableChunk] | None
        while (msg := await ch_in.recv(ctx)) is not None:
            # Convert the message back into a table chunk (releases the message).
            table = TableChunk.from_message(msg)

            # Accumulate the number of rows.
            total_num_rows[0] += table.table_view().num_rows()

            # The message is now empty since it was released.
            assert msg.empty()

            # Wrap the table chunk in a new message.
            msg = Message(table)

            # Forward the message to the output channel.
            await ch_out.send(ctx, msg)

        # `msg == None` indicates the channel is closed, i.e. we are done.
        # Before exiting, drain the output channel to close it gracefully.
        await ch_out.drain(ctx)

    # Nodes return None, so if we want an "output" value we can use either a closure
    # or an output parameter like `total_num_rows`.
    total_num_rows = [0]  # Wrap scalar in a list to make it mutable in-place.
    node2: PyNode = count_num_rows(
        ctx, ch_in=ch1, ch_out=ch2, total_num_rows=total_num_rows
    )

    # Node 3: consumer that pulls messages from the pipeline.
    # Like push_to_channel(), it returns a CppNode. It also returns a placeholder
    # object that will be populated with the pulled messages after execution.
    node3, out_messages = pull_from_channel(ctx, ch_in=ch2)

    # Run all nodes. This blocks until every node has completed.
    run_streaming_pipeline(
        nodes=(
            node1,
            node2,
            node3,
        ),
        py_executor=py_executor,
    )

    # Collect and verify results.
    expect = 0
    for msg in out_messages.release():
        table = TableChunk.from_message(msg).table_view()
        expect += table.num_rows()
    assert total_num_rows[0] == expect
    return total_num_rows[0]


if __name__ == "__main__":
    print(f"total_num_rows: {main()}")
