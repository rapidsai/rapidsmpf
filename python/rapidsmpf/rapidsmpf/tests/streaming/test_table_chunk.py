# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import cudf

from rapidsmpf.streaming.core.channel import Message
from rapidsmpf.streaming.cudf.table_chunk import TableChunk
from rapidsmpf.testing import assert_eq
from rapidsmpf.utils.cudf import cudf_to_pylibcudf_table

if TYPE_CHECKING:
    from rmm.pylibrmm.stream import Stream

    from rapidsmpf.streaming.core.context import Context


def test_from_pylibcudf_table(context: Context, stream: Stream) -> None:
    seq = 42
    expect = cudf_to_pylibcudf_table(cudf.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    table_chunk = TableChunk.from_pylibcudf_table(seq, expect, stream)
    assert table_chunk.sequence_number() == seq
    assert table_chunk.is_available()
    assert_eq(expect, table_chunk.table_view())

    # Message roundtrip check.
    assert_eq(expect, TableChunk.from_message(Message(table_chunk)).table_view())
