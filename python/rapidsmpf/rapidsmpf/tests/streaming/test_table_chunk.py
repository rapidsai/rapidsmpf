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
def test_from_pylibcudf_table(
    context: Context, stream: Stream, *, exclusive_view: bool
) -> None:
    seq = 42
    expect = random_table(1024)
    table_chunk = TableChunk.from_pylibcudf_table(
        expect, stream, exclusive_view=exclusive_view
    )
    assert is_equal_streams(table_chunk.stream, stream)
    assert table_chunk.is_available()
    assert table_chunk.is_spillable() == exclusive_view
    assert_eq(expect, table_chunk.table_view())

    # Message roundtrip check.
    msg = Message(seq, table_chunk)
    assert msg.sequence_number == seq
    assert msg.get_content_description() == ContentDescription(
        content_sizes={MemoryType.DEVICE: 1024, MemoryType.HOST: 0},
        spillable=exclusive_view,
    )
    table_chunk2 = TableChunk.from_message(msg)
    assert is_equal_streams(table_chunk2.stream, stream)
    assert table_chunk2.is_available()
    assert_eq(expect, table_chunk2.table_view())
