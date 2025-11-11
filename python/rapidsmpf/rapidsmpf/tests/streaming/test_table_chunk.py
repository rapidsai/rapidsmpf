# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import cupy
import pytest

import cudf

from rapidsmpf.buffer.buffer import MemoryType
from rapidsmpf.buffer.content_description import ContentDescription
from rapidsmpf.cuda_stream import is_equal_streams
from rapidsmpf.streaming.core.message import Message
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
        content_sizes={MemoryType.DEVICE: 1024, MemoryType.HOST: 0},
        spillable=exclusive_view,
    )

    # Make a copy of msg1 in host memory.
    assert msg1.copy_cost() == 1024
    [res, _] = context.br().reserve(MemoryType.HOST, 1024, allow_overbooking=True)
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
    [res, _] = context.br().reserve(MemoryType.DEVICE, 1024, allow_overbooking=True)
    msg3 = msg2.copy(res)
    assert res.size == 0

    # msg2 is on host and is not availabe
    table_chunk3 = TableChunk.from_message(msg2)
    assert is_equal_streams(table_chunk3.stream, stream)
    assert not table_chunk3.is_available()
    assert table_chunk3.make_available_cost() == 1024
    # but we can make its table available using `make_available()`.
    [res, _] = context.br().reserve(MemoryType.DEVICE, 1024, allow_overbooking=True)
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
    [res, _] = context.br().reserve(MemoryType.DEVICE, 0, allow_overbooking=True)
    table_chunk6 = table_chunk5.make_available(res)
    assert table_chunk6.is_available()
    assert table_chunk6.make_available_cost() == 0
    assert_eq(expect, table_chunk6.table_view())
