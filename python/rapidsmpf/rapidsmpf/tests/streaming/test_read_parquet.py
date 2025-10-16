# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import numpy as np
import pytest

import pylibcudf as plc

from rapidsmpf.streaming.core.channel import Channel
from rapidsmpf.streaming.core.leaf_node import pull_from_channel
from rapidsmpf.streaming.core.node import run_streaming_pipeline
from rapidsmpf.streaming.cudf.parquet import read_parquet
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

if TYPE_CHECKING:
    from typing import Literal

    from rapidsmpf.streaming.core.context import Context


@pytest.fixture(scope="module")
def source(
    tmp_path_factory: pytest.TempPathFactory,
) -> plc.io.SourceInfo:
    path = tmp_path_factory.mktemp("read_parquet")

    nrows = 10
    start = 0
    sources = []
    for i in range(10):
        table = plc.Table(
            [plc.Column.from_array(np.arange(start, start + nrows, dtype="int32"))]
        )
        # gaps in the column numbering we produce
        start += nrows + nrows // 2
        filename = path / f"{i:3d}.pq"
        sink = plc.io.SinkInfo([filename])
        options = plc.io.parquet.ParquetWriterOptions.builder(sink, table).build()
        plc.io.parquet.write_parquet(options)
        sources.append(filename)
    return plc.io.SourceInfo(sources)


@pytest.mark.parametrize(
    "skip_rows", ["none", 7, 19, 113], ids=lambda s: f"skip_rows_{s}"
)
@pytest.mark.parametrize("num_rows", ["all", 0, 3, 31, 83], ids=lambda s: f"nrows_{s}")
def test_read_parquet(
    context: Context,
    source: plc.io.SourceInfo,
    skip_rows: int | Literal["none"],
    num_rows: int | Literal["all"],
) -> None:
    ch = Channel[TableChunk]()

    options = plc.io.parquet.ParquetReaderOptions.builder(source).build()

    if skip_rows != "none":
        options.set_skip_rows(skip_rows)
    if num_rows != "all":
        options.set_num_rows(num_rows)
    expected = plc.io.parquet.read_parquet(options).tbl

    producer = read_parquet(context, ch, 4, options, 3)

    consumer, messages = pull_from_channel(context, ch)

    run_streaming_pipeline(nodes=[producer, consumer])

    chunks = [TableChunk.from_message(m) for m in messages.release()]
    for chunk in chunks:
        chunk.stream.synchronize()

    got = plc.concatenate.concatenate(
        [
            chunk.table_view()
            for chunk in sorted(chunks, key=operator.attrgetter("sequence_number"))
        ]
    )
    for chunk in chunks:
        chunk.stream.synchronize()

    assert got.num_rows() == expected.num_rows()
    assert got.num_columns() == expected.num_columns()
    assert got.num_columns() == 1

    all_equal = plc.reduce.reduce(
        plc.binaryop.binary_operation(
            got.columns()[0],
            expected.columns()[0],
            plc.binaryop.BinaryOperator.EQUAL,
            plc.DataType(plc.TypeId.BOOL8),
        ),
        plc.aggregation.all(),
        plc.DataType(plc.TypeId.BOOL8),
    )
    assert all_equal.to_py()


@pytest.mark.parametrize("num_tickets", [-1, 0])
def test_read_parquet_non_positive_throttle_throws(
    context: Context, source: plc.io.SourceInfo, num_tickets: int
) -> None:
    ch = Channel[TableChunk]()
    options = plc.io.parquet.ParquetReaderOptions.builder(source).build()

    producer = read_parquet(context, ch, num_tickets, options, 100)

    with pytest.raises(RuntimeError):
        run_streaming_pipeline(nodes=[producer])
