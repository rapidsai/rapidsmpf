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
from rapidsmpf.streaming.core.spillable_messages import SpillableMessages
from rapidsmpf.streaming.cudf.table_chunk import TableChunk
from rapidsmpf.testing import assert_eq
from rapidsmpf.utils.cudf import cudf_to_pylibcudf_table

if TYPE_CHECKING:
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

    # msg3 is on device (was created by copying the host msg2). During the copy this
    # is made available trivially.
    table_chunk5 = TableChunk.from_message(msg3)
    assert is_equal_streams(table_chunk5.stream, stream)
    assert table_chunk5.is_available()
    # and it cost no device memory to make available.
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
