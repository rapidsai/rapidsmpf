# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import cudf

from rapidsmpf.cuda_stream import is_equal_streams
from rapidsmpf.streaming.core.channel import Message
from rapidsmpf.streaming.cudf.table_chunk import TableChunk
from rapidsmpf.testing import assert_eq
from rapidsmpf.utils.cudf import cudf_to_pylibcudf_table

if TYPE_CHECKING:
    from rmm.pylibrmm.stream import Stream

    from rapidsmpf.streaming.core.context import Context


@pytest.mark.parametrize(
    "is_exclusive_view",
    [True, False],
)
def test_from_pylibcudf_table(
    context: Context, stream: Stream, *, is_exclusive_view: bool
) -> None:
    seq = 42
    expect = cudf_to_pylibcudf_table(cudf.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    table_chunk = TableChunk.from_pylibcudf_table(
        seq, expect, stream, is_exclusive_view=is_exclusive_view
    )
    assert table_chunk.sequence_number == seq
    assert is_equal_streams(table_chunk.stream, stream)
    assert table_chunk.is_available()
    assert table_chunk.is_spillable() == is_exclusive_view
    assert_eq(expect, table_chunk.table_view())

    # Message roundtrip check.
    table_chunk2 = TableChunk.from_message(Message(table_chunk))
    assert table_chunk2.sequence_number == seq
    assert is_equal_streams(table_chunk2.stream, stream)
    assert table_chunk2.is_available()
    assert_eq(expect, table_chunk2.table_view())
