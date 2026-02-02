# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import cupy
import pytest

import cudf

from rapidsmpf.cuda_stream import is_equal_streams
from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.memory.content_description import ContentDescription
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.core.node import define_py_node, run_streaming_pipeline
from rapidsmpf.streaming.core.spillable_messages import SpillableMessages
from rapidsmpf.streaming.cudf.table_chunk import (
    TableChunk,
    make_table_chunks_available_or_wait,
)
from rapidsmpf.testing import assert_eq
from rapidsmpf.utils.cudf import cudf_to_pylibcudf_table

if TYPE_CHECKING:
    from concurrent.futures import ThreadPoolExecutor

    import pylibcudf
    from rmm.pylibrmm.stream import Stream

    from rapidsmpf.streaming.core.context import Context


def random_table(nbytes: int) -> pylibcudf.Table:
    assert nbytes % 4 == 0
    return cudf_to_pylibcudf_table(
        cudf.DataFrame(
            {
                "data": cupy.random.random(nbytes // 4, dtype=cupy.float32),
            }
        )
    )


@pytest.mark.parametrize(
    "exclusive_view",
    [True, False],
)
def test_roundtrip(context: Context, stream: Stream, *, exclusive_view: bool) -> None:
    seq = 42
    expect = random_table(1024)
    table_chunk = TableChunk.from_pylibcudf_table(
        expect, stream, exclusive_view=exclusive_view
    )
    assert is_equal_streams(table_chunk.stream, stream)
    assert table_chunk.is_available()
    assert table_chunk.make_available_cost() == 0
    assert table_chunk.is_spillable() == exclusive_view
    assert_eq(expect, table_chunk.table_view())

    # Message roundtrip check.
    msg1 = Message(seq, table_chunk)
    assert msg1.sequence_number == seq
    assert msg1.get_content_description() == ContentDescription(
        content_sizes={
            MemoryType.DEVICE: 1024,
            MemoryType.PINNED_HOST: 0,
            MemoryType.HOST: 0,
        },
        spillable=exclusive_view,
    )

    # Make a copy of msg1 in host memory.
    assert msg1.copy_cost() == 1024
    res, _ = context.br().reserve(MemoryType.HOST, 1024, allow_overbooking=True)
    msg2 = msg1.copy(res)
    assert res.size == 0

    # msg1 is availabe
    table_chunk2 = TableChunk.from_message(msg1)
    assert is_equal_streams(table_chunk2.stream, stream)
    assert table_chunk2.is_available()
    assert table_chunk2.make_available_cost() == 0
    assert_eq(expect, table_chunk2.table_view())

    # Make a copy of msg2 back to device memory.
    assert msg2.copy_cost() == 1024
    res, _ = context.br().reserve(MemoryType.DEVICE, 1024, allow_overbooking=True)
    msg3 = msg2.copy(res)
    assert res.size == 0

    # msg2 is on host and is not availabe
    table_chunk3 = TableChunk.from_message(msg2)
    assert is_equal_streams(table_chunk3.stream, stream)
    assert not table_chunk3.is_available()
    assert table_chunk3.make_available_cost() == 1024
    # but we can make its table available using `make_available()`.
    res, _ = context.br().reserve(MemoryType.DEVICE, 1024, allow_overbooking=True)
    table_chunk4 = table_chunk3.make_available(res)
    assert is_equal_streams(table_chunk4.stream, stream)
    assert table_chunk4.is_available()
    assert table_chunk4.make_available_cost() == 0
    assert_eq(expect, table_chunk4.table_view())

    # msg3 is on device but not availabe.
    table_chunk5 = TableChunk.from_message(msg3)
    assert is_equal_streams(table_chunk5.stream, stream)
    assert not table_chunk5.is_available()
    # but it cost no device memory to make available.
    assert table_chunk5.make_available_cost() == 0
    res, _ = context.br().reserve(MemoryType.DEVICE, 0, allow_overbooking=True)
    table_chunk6 = table_chunk5.make_available(res)
    assert table_chunk6.is_available()
    assert table_chunk6.make_available_cost() == 0
    assert_eq(expect, table_chunk6.table_view())


def test_copy_roundtrip(context: Context, stream: Stream) -> None:
    for nrows, ncols in [(1, 1), (1000, 100), (1, 1000)]:
        expect = cudf_to_pylibcudf_table(
            cudf.DataFrame(
                {
                    f"{name}": cupy.random.random(nrows, dtype=cupy.float32)
                    for name in range(ncols)
                }
            )
        )

        tbl1 = TableChunk.from_pylibcudf_table(expect, stream, exclusive_view=True)
        res, _ = context.br().reserve(
            MemoryType.HOST,
            tbl1.data_alloc_size(MemoryType.DEVICE),
            allow_overbooking=True,
        )
        tbl2 = tbl1.copy(res)
        res, _ = context.br().reserve(
            MemoryType.DEVICE, tbl2.make_available_cost(), allow_overbooking=True
        )
        tbl3 = tbl2.make_available(res)
        assert_eq(expect, tbl3.table_view())


def test_spillable_messages(context: Context, stream: Stream) -> None:
    seq = 42
    df1 = random_table(1024)
    df2 = random_table(2048)

    sm = SpillableMessages()
    sm.insert(
        Message(seq, TableChunk.from_pylibcudf_table(df1, stream, exclusive_view=True))
    )
    assert sm.get_content_descriptions() == {
        0: ContentDescription(
            content_sizes={
                MemoryType.DEVICE: 1024,
                MemoryType.PINNED_HOST: 0,
                MemoryType.HOST: 0,
            },
            spillable=True,
        )
    }
    sm.insert(
        Message(seq, TableChunk.from_pylibcudf_table(df2, stream, exclusive_view=False))
    )
    assert sm.get_content_descriptions() == {
        0: ContentDescription(
            content_sizes={
                MemoryType.DEVICE: 1024,
                MemoryType.PINNED_HOST: 0,
                MemoryType.HOST: 0,
            },
            spillable=True,
        ),
        1: ContentDescription(
            content_sizes={
                MemoryType.DEVICE: 2048,
                MemoryType.PINNED_HOST: 0,
                MemoryType.HOST: 0,
            },
            spillable=False,
        ),
    }
    assert sm.spill(mid=0, br=context.br()) == 1024
    assert sm.get_content_descriptions() == {
        0: ContentDescription(
            content_sizes={
                MemoryType.DEVICE: 0,
                MemoryType.PINNED_HOST: 0,
                MemoryType.HOST: 1024,
            },
            spillable=True,
        ),
        1: ContentDescription(
            content_sizes={
                MemoryType.DEVICE: 2048,
                MemoryType.PINNED_HOST: 0,
                MemoryType.HOST: 0,
            },
            spillable=False,
        ),
    }
    assert sm.spill(mid=1, br=context.br()) == 0
    assert sm.get_content_descriptions() == {
        0: ContentDescription(
            content_sizes={
                MemoryType.DEVICE: 0,
                MemoryType.PINNED_HOST: 0,
                MemoryType.HOST: 1024,
            },
            spillable=True,
        ),
        1: ContentDescription(
            content_sizes={
                MemoryType.DEVICE: 2048,
                MemoryType.PINNED_HOST: 0,
                MemoryType.HOST: 0,
            },
            spillable=False,
        ),
    }

    # Extract, make available, and check table chunk 1.
    df1_got = TableChunk.from_message(sm.extract(mid=0))
    res, _ = context.br().reserve(
        MemoryType.DEVICE, df1_got.make_available_cost(), allow_overbooking=True
    )
    df1_got = df1_got.make_available(res)
    assert_eq(df1, df1_got.table_view())

    with pytest.raises(IndexError, match="Invalid key"):
        sm.extract(mid=0)

    df2_got = TableChunk.from_message(sm.extract(mid=1))
    df2_got = df2_got.make_available_and_spill(context.br(), allow_overbooking=True)
    assert_eq(df2, df2_got.table_view())
    assert sm.get_content_descriptions() == {}


def test_spillable_messages_by_context(context: Context, stream: Stream) -> None:
    seq = 42
    expect = random_table(1024)

    mid = context.spillable_messages().insert(
        Message(
            seq, TableChunk.from_pylibcudf_table(expect, stream, exclusive_view=True)
        )
    )
    assert context.spillable_messages().get_content_descriptions() == {
        0: ContentDescription(
            content_sizes={
                MemoryType.DEVICE: 1024,
                MemoryType.PINNED_HOST: 0,
                MemoryType.HOST: 0,
            },
            spillable=True,
        )
    }
    got = TableChunk.from_message(context.spillable_messages().extract(mid=mid))
    assert_eq(expect, got.table_view())


def test_make_available_or_wait_already_available(
    context: Context, stream: Stream, py_executor: ThreadPoolExecutor
) -> None:
    expect = random_table(1024)
    chunk = TableChunk.from_pylibcudf_table(expect, stream, exclusive_view=True)
    result_holder: list[TableChunk] = []

    @define_py_node()
    async def test_node(ctx: Context) -> None:
        result = await chunk.make_available_or_wait(ctx, net_memory_delta=0)
        result_holder.append(result)

    run_streaming_pipeline(nodes=[test_node(context)], py_executor=py_executor)
    assert_eq(expect, result_holder[0].table_view())


@pytest.mark.parametrize("net_memory_delta", [0, 512])
def test_make_available_or_wait_from_host(
    context: Context,
    stream: Stream,
    py_executor: ThreadPoolExecutor,
    *,
    net_memory_delta: int,
) -> None:
    expect = random_table(1024)
    device_chunk = TableChunk.from_pylibcudf_table(expect, stream, exclusive_view=True)
    res, _ = context.br().reserve(
        MemoryType.HOST,
        device_chunk.data_alloc_size(MemoryType.DEVICE),
        allow_overbooking=True,
    )
    host_chunk = device_chunk.copy(res)
    result_holder: list[TableChunk] = []

    @define_py_node()
    async def test_node(ctx: Context) -> None:
        result = await host_chunk.make_available_or_wait(
            ctx, net_memory_delta=net_memory_delta
        )
        result_holder.append(result)

    run_streaming_pipeline(nodes=[test_node(context)], py_executor=py_executor)
    assert_eq(expect, result_holder[0].table_view())


def test_data_alloc_size(context: Context, stream: Stream) -> None:
    # Create a table chunk on device memory.
    expect = random_table(1024)
    device_chunk = TableChunk.from_pylibcudf_table(expect, stream, exclusive_view=True)

    # Check device memory size.
    assert device_chunk.data_alloc_size(MemoryType.DEVICE) == 1024
    assert device_chunk.data_alloc_size(MemoryType.HOST) == 0
    assert device_chunk.data_alloc_size(MemoryType.PINNED_HOST) == 0

    # Check that None returns the total across all memory types.
    total_size = device_chunk.data_alloc_size(None)
    assert total_size == 1024
    assert total_size == (
        device_chunk.data_alloc_size(MemoryType.DEVICE)
        + device_chunk.data_alloc_size(MemoryType.HOST)
        + device_chunk.data_alloc_size(MemoryType.PINNED_HOST)
    )

    # Check that calling without arguments (default None) works the same.
    assert device_chunk.data_alloc_size() == 1024
    assert device_chunk.data_alloc_size() == device_chunk.data_alloc_size(None)

    # Copy to host memory and verify memory distribution.
    res, _ = context.br().reserve(
        MemoryType.HOST,
        device_chunk.data_alloc_size(MemoryType.DEVICE),
        allow_overbooking=True,
    )
    host_chunk = device_chunk.copy(res)

    assert host_chunk.data_alloc_size(MemoryType.DEVICE) == 0
    assert host_chunk.data_alloc_size(MemoryType.HOST) == 1024
    assert host_chunk.data_alloc_size(MemoryType.PINNED_HOST) == 0

    # Check that None still returns the correct total.
    total_size = host_chunk.data_alloc_size(None)
    assert total_size == 1024
    assert total_size == (
        host_chunk.data_alloc_size(MemoryType.DEVICE)
        + host_chunk.data_alloc_size(MemoryType.HOST)
        + host_chunk.data_alloc_size(MemoryType.PINNED_HOST)
    )

    # Verify default parameter works after copy too.
    assert host_chunk.data_alloc_size() == 1024
    assert host_chunk.data_alloc_size() == host_chunk.data_alloc_size(None)


@pytest.mark.parametrize("chunk_location", ["device", "host"])
def test_make_table_chunks_available_or_wait_single_chunk(
    context: Context,
    stream: Stream,
    py_executor: ThreadPoolExecutor,
    *,
    chunk_location: str,
) -> None:
    expect = random_table(1024)
    device_chunk = TableChunk.from_pylibcudf_table(expect, stream, exclusive_view=True)

    if chunk_location == "host":
        res_holder, _ = context.br().reserve(
            MemoryType.HOST,
            device_chunk.data_alloc_size(MemoryType.DEVICE),
            allow_overbooking=True,
        )
        chunk = device_chunk.copy(res_holder)
    else:
        chunk = device_chunk

    result_holder: list[tuple] = []

    @define_py_node()
    async def test_node(ctx: Context) -> None:
        result_chunk, res = await make_table_chunks_available_or_wait(
            ctx, chunk, reserve_extra=0, net_memory_delta=0
        )
        result_holder.append((result_chunk, res))

    run_streaming_pipeline(nodes=[test_node(context)], py_executor=py_executor)
    chunk, res = result_holder[0]
    assert chunk.is_available()
    assert_eq(expect, chunk.table_view())
    # Reservation should be consumed by making the chunk available.
    assert res.size == 0


@pytest.mark.parametrize("num_chunks", [2, 3, 5])
def test_make_table_chunks_available_or_wait_multiple_chunks(
    context: Context,
    stream: Stream,
    py_executor: ThreadPoolExecutor,
    *,
    num_chunks: int,
) -> None:
    # Create multiple chunks with different sizes.
    sizes = [1024, 2048, 512, 768, 1536][:num_chunks]
    expects = [random_table(size) for size in sizes]

    # Create host chunks.
    device_chunks = [
        TableChunk.from_pylibcudf_table(expect, stream, exclusive_view=True)
        for expect in expects
    ]

    host_chunks = []
    for device_chunk in device_chunks:
        res, _ = context.br().reserve(
            MemoryType.HOST,
            device_chunk.data_alloc_size(MemoryType.DEVICE),
            allow_overbooking=True,
        )
        host_chunks.append(device_chunk.copy(res))

    result_holder: list[tuple] = []

    @define_py_node()
    async def test_node(ctx: Context) -> None:
        chunks, res = await make_table_chunks_available_or_wait(
            ctx,
            host_chunks,
            reserve_extra=0,
            net_memory_delta=0,
        )
        result_holder.append((chunks, res))

    run_streaming_pipeline(nodes=[test_node(context)], py_executor=py_executor)
    chunks, res = result_holder[0]
    assert len(chunks) == num_chunks
    assert all(chunk.is_available() for chunk in chunks)
    for i, expect in enumerate(expects):
        assert_eq(expect, chunks[i].table_view())
    # Reservation should be consumed.
    assert res.size == 0


@pytest.mark.parametrize("reserve_extra", [0, 512, 1024])
def test_make_table_chunks_available_or_wait_with_reserve_extra(
    context: Context,
    stream: Stream,
    py_executor: ThreadPoolExecutor,
    *,
    reserve_extra: int,
) -> None:
    expect = random_table(1024)
    device_chunk = TableChunk.from_pylibcudf_table(expect, stream, exclusive_view=True)
    res_holder, _ = context.br().reserve(
        MemoryType.HOST,
        device_chunk.data_alloc_size(MemoryType.DEVICE),
        allow_overbooking=True,
    )
    host_chunk = device_chunk.copy(res_holder)
    result_holder: list[tuple] = []

    @define_py_node()
    async def test_node(ctx: Context) -> None:
        chunk, res = await make_table_chunks_available_or_wait(
            ctx,
            host_chunk,
            reserve_extra=reserve_extra,
            net_memory_delta=0,
        )
        result_holder.append((chunk, res))

    run_streaming_pipeline(nodes=[test_node(context)], py_executor=py_executor)
    chunk, res = result_holder[0]
    assert chunk.is_available()
    assert_eq(expect, chunk.table_view())
    # Reservation should have reserve_extra bytes remaining.
    assert res.size == reserve_extra


@pytest.mark.parametrize("net_memory_delta", [-1024, 0, 512, 2048])
def test_make_table_chunks_available_or_wait_with_net_memory_delta(
    context: Context,
    stream: Stream,
    py_executor: ThreadPoolExecutor,
    *,
    net_memory_delta: int,
) -> None:
    expect = random_table(1024)
    device_chunk = TableChunk.from_pylibcudf_table(expect, stream, exclusive_view=True)
    res_holder, _ = context.br().reserve(
        MemoryType.HOST,
        device_chunk.data_alloc_size(MemoryType.DEVICE),
        allow_overbooking=True,
    )
    host_chunk = device_chunk.copy(res_holder)
    result_holder: list[tuple] = []

    @define_py_node()
    async def test_node(ctx: Context) -> None:
        chunk, res = await make_table_chunks_available_or_wait(
            ctx,
            host_chunk,
            reserve_extra=0,
            net_memory_delta=net_memory_delta,
        )
        result_holder.append((chunk, res))

    run_streaming_pipeline(nodes=[test_node(context)], py_executor=py_executor)
    chunk, res = result_holder[0]
    assert chunk.is_available()
    assert_eq(expect, chunk.table_view())
    # Reservation should be consumed.
    assert res.size == 0


def test_make_table_chunks_available_or_wait_mixed_availability(
    context: Context, stream: Stream, py_executor: ThreadPoolExecutor
) -> None:
    expect1 = random_table(1024)
    expect2 = random_table(2048)

    # First chunk is already available on device.
    available_chunk = TableChunk.from_pylibcudf_table(
        expect1, stream, exclusive_view=True
    )

    # Second chunk is on host memory.
    device_chunk2 = TableChunk.from_pylibcudf_table(
        expect2, stream, exclusive_view=True
    )
    res2, _ = context.br().reserve(
        MemoryType.HOST,
        device_chunk2.data_alloc_size(MemoryType.DEVICE),
        allow_overbooking=True,
    )
    host_chunk = device_chunk2.copy(res2)
    result_holder: list[tuple] = []

    @define_py_node()
    async def test_node(ctx: Context) -> None:
        chunks, res = await make_table_chunks_available_or_wait(
            ctx,
            [available_chunk, host_chunk],
            reserve_extra=0,
            net_memory_delta=0,
        )
        result_holder.append((chunks, res))

    run_streaming_pipeline(nodes=[test_node(context)], py_executor=py_executor)
    chunks, res = result_holder[0]
    assert len(chunks) == 2
    assert all(chunk.is_available() for chunk in chunks)
    assert_eq(expect1, chunks[0].table_view())
    assert_eq(expect2, chunks[1].table_view())
    # Only the host chunk required device memory.
    assert res.size == 0


@pytest.mark.parametrize("allow_overbooking", [True, False, None])
def test_make_table_chunks_available_or_wait_overbooking_parameter(
    context: Context,
    stream: Stream,
    py_executor: ThreadPoolExecutor,
    *,
    allow_overbooking: bool | None,
) -> None:
    expect = random_table(1024)
    device_chunk = TableChunk.from_pylibcudf_table(expect, stream, exclusive_view=True)
    res_holder, _ = context.br().reserve(
        MemoryType.HOST,
        device_chunk.data_alloc_size(MemoryType.DEVICE),
        allow_overbooking=True,
    )
    host_chunk = device_chunk.copy(res_holder)
    result_holder: list[tuple] = []

    @define_py_node()
    async def test_node(ctx: Context) -> None:
        chunk, res = await make_table_chunks_available_or_wait(
            ctx,
            host_chunk,
            reserve_extra=0,
            net_memory_delta=0,
            allow_overbooking=allow_overbooking,
        )
        result_holder.append((chunk, res))

    run_streaming_pipeline(nodes=[test_node(context)], py_executor=py_executor)
    chunk, res = result_holder[0]
    assert chunk.is_available()
    assert_eq(expect, chunk.table_view())
    # Reservation should be consumed.
    assert res.size == 0
