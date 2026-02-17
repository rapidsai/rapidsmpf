# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import pylibcudf as plc

from rapidsmpf.streaming.core.leaf_node import pull_from_channel, push_to_channel
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.core.node import run_actor_graph
from rapidsmpf.streaming.cudf.bloom_filter import BloomFilter
from rapidsmpf.streaming.cudf.table_chunk import TableChunk
from rapidsmpf.testing import assert_eq

if TYPE_CHECKING:
    from rmm.pylibrmm.stream import Stream

    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context
    from rapidsmpf.streaming.cudf.bloom_filter import BloomFilterChunk


def make_table(values: np.ndarray, stream: Stream) -> TableChunk:
    table = plc.Table([plc.Column.from_array(values, stream=stream)])
    return TableChunk.from_pylibcudf_table(table, stream, exclusive_view=True)


def run_bloom_filter_pipeline(
    context: Context,
    build_table: TableChunk,
    probe_table: TableChunk,
    *,
    seed: int = 42,
    l2size: int = 1 << 20,
) -> list[Message]:
    bloom = BloomFilter(
        context,
        seed=seed,
        num_filter_blocks=BloomFilter.fitting_num_blocks(l2size),
    )

    build_msg = Message(0, build_table)
    probe_msg = Message(0, probe_table)

    ch_build: Channel[TableChunk] = context.create_channel()
    ch_probe: Channel[TableChunk] = context.create_channel()
    ch_filter: Channel[BloomFilterChunk] = context.create_channel()
    ch_out: Channel[TableChunk] = context.create_channel()

    nodes = [
        push_to_channel(context, ch_build, [build_msg]),
        push_to_channel(context, ch_probe, [probe_msg]),
        bloom.build(ch_in=ch_build, ch_out=ch_filter, tag=0),
        bloom.apply(
            bloom_filter=ch_filter,
            ch_in=ch_probe,
            ch_out=ch_out,
            keys=(0,),
        ),
    ]
    pull_node, deferred = pull_from_channel(context, ch_out)
    nodes.append(pull_node)
    run_actor_graph(nodes=nodes)
    return deferred.release()


def test_bloom_filter_roundtrip(context: Context) -> None:
    if context.comm().nranks != 1:
        pytest.skip("Only support single-rank runs")

    stream = context.get_stream_from_pool()
    values = np.arange(10, dtype=np.int32)
    build_table = make_table(values, stream=stream)
    probe_table = make_table(values, stream=stream)
    messages = run_bloom_filter_pipeline(context, build_table, probe_table)
    assert len(messages) == 1

    result = TableChunk.from_message(messages[0])
    expected = plc.Table([plc.Column.from_array(values, stream=result.stream)])
    result.stream.synchronize()
    assert_eq(result.table_view(), expected)


def test_bloom_filter_empty_build_filters_all(context: Context) -> None:
    if context.comm().nranks != 1:
        pytest.skip("Only support single-rank runs")

    stream = context.get_stream_from_pool()
    build_table = make_table(np.array([], dtype=np.int32), stream=stream)
    probe_table = make_table(np.arange(5, dtype=np.int32), stream=stream)
    messages = run_bloom_filter_pipeline(context, build_table, probe_table)
    assert len(messages) == 1

    result = TableChunk.from_message(messages[0])
    expected = plc.Table(
        [plc.Column.from_array(np.array([], dtype=np.int32), stream=result.stream)]
    )
    result.stream.synchronize()
    assert_eq(result.table_view(), expected)
