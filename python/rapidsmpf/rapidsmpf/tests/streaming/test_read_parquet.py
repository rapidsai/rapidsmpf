# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np
import pytest

import pylibcudf as plc

from rapidsmpf.streaming.core.leaf_node import pull_from_channel
from rapidsmpf.streaming.core.node import run_streaming_pipeline
from rapidsmpf.streaming.cudf.parquet import Filter, read_parquet
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

if TYPE_CHECKING:
    from typing import Literal

    from rmm.pylibrmm.stream import Stream

    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context
    from rapidsmpf.streaming.core.node import CppNode


def _create_parquet_files(
    tmp_path_factory: pytest.TempPathFactory, num_files: int, nrows_per_file: int = 10
) -> plc.io.SourceInfo:
    """Helper to create parquet files for testing."""
    path = tmp_path_factory.mktemp("read_parquet")

    start = 0
    sources = []
    for i in range(num_files):
        table = plc.Table(
            [
                plc.Column.from_array(
                    np.arange(start, start + nrows_per_file, dtype="int32")
                )
            ]
        )
        # gaps in the column numbering we produce
        start += nrows_per_file + nrows_per_file // 2
        filename = path / f"{i:3d}.pq"
        sink = plc.io.SinkInfo([filename])
        options = plc.io.parquet.ParquetWriterOptions.builder(sink, table).build()
        plc.io.parquet.write_parquet(options)
        sources.append(filename)
    return plc.io.SourceInfo(sources)


@pytest.fixture(scope="module")
def source(
    tmp_path_factory: pytest.TempPathFactory,
) -> plc.io.SourceInfo:
    return _create_parquet_files(tmp_path_factory, num_files=10)


@pytest.fixture(scope="module")
def single_file_source(
    tmp_path_factory: pytest.TempPathFactory,
) -> plc.io.SourceInfo:
    """Single file source for testing fewer-files-than-ranks scenario."""
    return _create_parquet_files(tmp_path_factory, num_files=1, nrows_per_file=100)


def make_filter(stream: Stream) -> plc.expressions.Expression:
    return plc.expressions.Operation(
        plc.expressions.ASTOperator.LESS,
        plc.expressions.ColumnReference(0),
        plc.expressions.Literal(
            plc.Scalar.from_py(15, dtype=plc.DataType(plc.TypeId.INT32), stream=stream)
        ),
    )


def make_producer(
    context: Context,
    ch: Channel[TableChunk],
    options: plc.io.parquet.ParquetReaderOptions,
    *,
    use_filter: bool,
) -> CppNode:
    if use_filter:
        fstream = context.get_stream_from_pool()
        return read_parquet(
            context, ch, 4, options, 3, Filter(fstream, make_filter(fstream))
        )
    else:
        return read_parquet(context, ch, 4, options, 3)


def get_expected(
    ctx: Context,
    source: plc.io.SourceInfo,
    skip_rows: int | Literal["none"],
    num_rows: int | Literal["all"],
    *,
    use_filter: bool,
) -> plc.Table:
    options = plc.io.parquet.ParquetReaderOptions.builder(source).build()

    if skip_rows != "none":
        options.set_skip_rows(skip_rows)
    if num_rows != "all":
        options.set_num_rows(num_rows)
    if use_filter:
        fstream = ctx.get_stream_from_pool()
        filter = make_filter(fstream)
        fstream.synchronize()
        options.set_filter(filter)

    expected = plc.io.parquet.read_parquet(options).tbl

    if use_filter:
        fstream.synchronize()
    return expected


@pytest.mark.parametrize(
    "skip_rows", ["none", 7, 19, 113], ids=lambda s: f"skip_rows_{s}"
)
@pytest.mark.parametrize("num_rows", ["all", 0, 3, 31, 83], ids=lambda s: f"nrows_{s}")
@pytest.mark.parametrize("use_filter", [False, True])
def test_read_parquet(
    context: Context,
    source: plc.io.SourceInfo,
    skip_rows: int | Literal["none"],
    num_rows: int | Literal["all"],
    use_filter: bool,  # noqa: FBT001
) -> None:
    ch: Channel[TableChunk] = context.create_channel()

    options = plc.io.parquet.ParquetReaderOptions.builder(source).build()

    if skip_rows != "none":
        options.set_skip_rows(skip_rows)
    if num_rows != "all":
        options.set_num_rows(num_rows)

    producer = make_producer(context, ch, options, use_filter=use_filter)

    consumer, deferred_messages = pull_from_channel(context, ch)

    run_streaming_pipeline(nodes=[producer, consumer])

    messages = deferred_messages.release()
    assert all(
        m1.sequence_number < m2.sequence_number
        for m1, m2 in itertools.pairwise(messages)
    )
    chunks = [TableChunk.from_message(m) for m in messages]
    for chunk in chunks:
        chunk.stream.synchronize()

    got = plc.concatenate.concatenate([chunk.table_view() for chunk in chunks])
    for chunk in chunks:
        chunk.stream.synchronize()

    expected = get_expected(context, source, skip_rows, num_rows, use_filter=use_filter)

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


def test_read_parquet_single_file(
    context: Context,
    single_file_source: plc.io.SourceInfo,
) -> None:
    """Test reading a single file, which exercises the fewer-files-than-ranks code path."""
    ch: Channel[TableChunk] = context.create_channel()

    options = plc.io.parquet.ParquetReaderOptions.builder(single_file_source).build()

    producer = make_producer(context, ch, options, use_filter=False)

    consumer, deferred_messages = pull_from_channel(context, ch)

    run_streaming_pipeline(nodes=[producer, consumer])

    messages = deferred_messages.release()
    assert all(
        m1.sequence_number < m2.sequence_number
        for m1, m2 in itertools.pairwise(messages)
    )
    chunks = [TableChunk.from_message(m) for m in messages]
    for chunk in chunks:
        chunk.stream.synchronize()

    got = plc.concatenate.concatenate([chunk.table_view() for chunk in chunks])
    for chunk in chunks:
        chunk.stream.synchronize()

    expected = get_expected(
        context, single_file_source, skip_rows="none", num_rows="all", use_filter=False
    )

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
