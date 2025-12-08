# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: D103

"""Example by hand implementation of a derivation of TPC-H Q9."""

from __future__ import annotations

import itertools
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import click
import nvtx

import pylibcudf as plc
import rmm
from pylibcudf.experimental._join_streams import join_streams

from rapidsmpf.buffer.resource import BufferResource
from rapidsmpf.communicator.single import new_communicator as single_process_comm
from rapidsmpf.config import Options, get_environment_variables
from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor
from rapidsmpf.streaming.core.channel import Channel
from rapidsmpf.streaming.core.context import Context
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.core.node import define_py_node, run_streaming_pipeline
from rapidsmpf.streaming.cudf.parquet import read_parquet
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rapidsmpf.streaming.core.node import CppNode, PyNode


def get_streaming_context(num_streaming_threads: int) -> Context:
    env = get_environment_variables()
    env["num_streaming_threads"] = str(num_streaming_threads)
    options = Options(env)
    comm = single_process_comm(options)
    # TODO: multi-GPU, memory limiter, spilling. Need to expose TableChunk::make_available.
    mr = RmmResourceAdaptor(rmm.mr.CudaAsyncMemoryResource())
    # TODO: explicit memory resources. Need to expose device_mr to python.
    br = BufferResource(mr)
    # Note: this must be done even if we use the br's memory resource
    # everywhere so that cudf uses this MR for internal allocations.
    rmm.mr.set_current_device_resource(mr)
    return Context(comm, br, options)


def reader_options(
    files: Sequence[str], columns: list[str]
) -> plc.io.parquet.ParquetReaderOptions:
    source = plc.io.SourceInfo(files)
    options = plc.io.parquet.ParquetReaderOptions.builder(source).build()
    options.set_columns(columns)
    return options


def read_lineitem(
    ctx: Context,
    files: Sequence[str],
    num_producers: int,
    num_rows_per_chunk: int,
    ch: Channel[TableChunk],
) -> CppNode | PyNode:
    columns = [
        "l_discount",  # 0
        "l_extendedprice",  # 1
        "l_orderkey",  # 2
        "l_partkey",  # 3
        "l_quantity",  # 4
        "l_suppkey",
    ]
    return read_parquet(
        ctx, ch, num_producers, reader_options(files, columns), num_rows_per_chunk
    )


def read_nation(
    ctx: Context,
    files: Sequence[str],
    num_rows_per_chunk: int,
    ch: Channel[TableChunk],
) -> CppNode | PyNode:
    columns = [
        "n_name",  # 0
        "n_nationkey",  # 1
    ]
    return read_parquet(ctx, ch, 1, reader_options(files, columns), num_rows_per_chunk)


def read_orders(
    ctx: Context,
    files: Sequence[str],
    num_producers: int,
    num_rows_per_chunk: int,
    ch: Channel[TableChunk],
) -> CppNode | PyNode:
    columns = [
        "o_orderdate",  # 0
        "o_orderkey",  # 1
    ]
    return read_parquet(
        ctx, ch, num_producers, reader_options(files, columns), num_rows_per_chunk
    )


def read_part(
    ctx: Context,
    files: Sequence[str],
    num_producers: int,
    num_rows_per_chunk: int,
    ch: Channel[TableChunk],
) -> CppNode | PyNode:
    columns = [
        "p_partkey",  # 0
        "p_name",  # 1
    ]
    return read_parquet(
        ctx, ch, num_producers, reader_options(files, columns), num_rows_per_chunk
    )


def read_partsupp(
    ctx: Context,
    files: Sequence[str],
    num_producers: int,
    num_rows_per_chunk: int,
    ch: Channel[TableChunk],
) -> CppNode | PyNode:
    columns = [
        "ps_partkey",  # 0
        "ps_suppkey",  # 1
        "ps_supplycost",  # 2
    ]
    return read_parquet(
        ctx, ch, num_producers, reader_options(files, columns), num_rows_per_chunk
    )


def read_supplier(
    ctx: Context,
    files: Sequence[str],
    num_producers: int,
    num_rows_per_chunk: int,
    ch: Channel[TableChunk],
) -> CppNode | PyNode:
    columns = [
        "s_nationkey",  # 0
        "s_suppkey",  # 1
    ]
    return read_parquet(
        ctx, ch, num_producers, reader_options(files, columns), num_rows_per_chunk
    )


@define_py_node()
async def filter_part(
    ctx: Context, ch_in: Channel[TableChunk], ch_out: Channel[TableChunk]
) -> None:
    while (msg := await ch_in.recv(ctx)) is not None:
        chunk = TableChunk.from_message(msg)
        stream = chunk.stream
        table = chunk.table_view()
        target = plc.Scalar.from_py("green", stream=stream)
        chunk = TableChunk.from_pylibcudf_table(
            plc.stream_compaction.apply_boolean_mask(
                plc.Table(table.columns()[:1]),
                plc.strings.find.contains(table.columns()[1], target, stream),
                stream,
            ),
            stream,
            exclusive_view=True,
        )
        await ch_out.send(ctx, Message(msg.sequence_number, chunk))
    await ch_out.drain(ctx)


@define_py_node()
async def select_columns(
    ctx: Context, ch_in: Channel[TableChunk], ch_out: Channel[TableChunk]
) -> None:
    while (msg := await ch_in.recv(ctx)) is not None:
        chunk = TableChunk.from_message(msg)
        stream = chunk.stream
        columns = chunk.table_view().columns()
        name, supplycost, discount, extendedprice, quantity, orderdate = columns
        revenue_type = supplycost.type()  # float64
        orderdate = plc.datetime.extract_datetime_component(
            orderdate, plc.datetime.DatetimeComponent.YEAR, stream
        )
        revenue = plc.transform.transform(
            [discount, extendedprice, supplycost, quantity],
            """
            static __device__ void calculate_amount(
                double *amount, double discount, double extprice, double supplycost, double quantity
            ) {
                *amount = extprice * (1 - discount) - supplycost * quantity;
            }
            """,
            revenue_type,
            False,  # noqa: FBT003
            plc.types.NullAware.NO,
            stream,
        )
        await ch_out.send(
            ctx,
            Message(
                msg.sequence_number,
                TableChunk.from_pylibcudf_table(
                    plc.Table([name, orderdate, revenue]),
                    stream,
                    exclusive_view=True,
                ),
            ),
        )
    await ch_out.drain(ctx)


@define_py_node()
async def broadcast_join(
    ctx: Context,
    left_ch: Channel[TableChunk],
    right_ch: Channel[TableChunk],
    ch_out: Channel[TableChunk],
    left_on: Sequence[int],
    right_on: Sequence[int],
    *,
    keep_keys: bool,
) -> None:
    left_tables: list[TableChunk] = []
    chunk_streams = set()
    while (msg := await left_ch.recv(ctx)) is not None:
        left_tables.append(TableChunk.from_message(msg))
        chunk_streams.add(left_tables[-1].stream)
    build_stream = ctx.get_stream_from_pool()
    join_streams(list(chunk_streams), build_stream)
    if len(left_tables) == 1:
        left = left_tables[0].table_view().columns()
    else:
        left = plc.concatenate.concatenate(
            [t.table_view() for t in left_tables], build_stream
        ).columns()
    left_keys = plc.Table([left[i] for i in left_on])
    if keep_keys:
        left_carrier = plc.Table(left)
    else:
        left_carrier = plc.Table([c for i, c in enumerate(left) if i not in left_on])
    for s in chunk_streams:
        join_streams([build_stream], s)
    del left_tables
    sequence_number = 0
    chunk_streams.clear()
    while (msg := await right_ch.recv(ctx)) is not None:
        chunk = TableChunk.from_message(msg)
        chunk_streams.add(chunk.stream)
        join_streams([build_stream], chunk.stream)
        # Safe to access left_carrier on chunk.stream
        right_columns = chunk.table_view().columns()
        right_keys = plc.Table([right_columns[i] for i in right_on])
        right_carrier = plc.Table(
            [c for i, c in enumerate(right_columns) if i not in right_on]
        )
        left, right = plc.join.inner_join(
            left_keys, right_keys, plc.types.NullEquality.UNEQUAL, chunk.stream
        )
        left = plc.copying.gather(
            left_carrier,
            left,
            plc.copying.OutOfBoundsPolicy.DONT_CHECK,
            chunk.stream,
        )
        right = plc.copying.gather(
            right_carrier,
            right,
            plc.copying.OutOfBoundsPolicy.DONT_CHECK,
            chunk.stream,
        )
        await ch_out.send(
            ctx,
            Message(
                sequence_number,
                TableChunk.from_pylibcudf_table(
                    plc.Table([*left.columns(), *right.columns()]),
                    chunk.stream,
                    exclusive_view=True,
                ),
            ),
        )
        sequence_number += 1
    # Ensure left_carrier and keys are deallocated after table chunks are produced
    for s in chunk_streams:
        join_streams([build_stream], s)
    await ch_out.drain(ctx)


@define_py_node()
async def chunkwise_groupby_agg(
    ctx: Context, ch_in: Channel[TableChunk], ch_out: Channel[TableChunk]
) -> None:
    sequence = 0
    while (msg := await ch_in.recv(ctx)) is not None:
        chunk = TableChunk.from_message(msg)
        name, date, revenue = chunk.table_view().columns()
        stream = chunk.stream
        grouper = plc.groupby.GroupBy(
            plc.Table([name, date]),
            plc.types.NullPolicy.EXCLUDE,
            plc.types.Sorted.NO,
        )
        reqs = [plc.groupby.GroupByRequest(revenue, [plc.aggregation.sum()])]
        (keys, results) = grouper.aggregate(reqs, stream)
        del chunk, name, date, revenue
        await ch_out.send(
            ctx,
            Message(
                sequence,
                TableChunk.from_pylibcudf_table(
                    plc.Table(
                        [
                            *keys.columns(),
                            *itertools.chain.from_iterable(
                                r.columns() for r in results
                            ),
                        ]
                    ),
                    stream,
                    exclusive_view=True,
                ),
            ),
        )
        sequence += 1
    await ch_out.drain(ctx)


@define_py_node()
async def concatenate(
    ctx: Context, ch_in: Channel[TableChunk], ch_out: Channel[TableChunk]
) -> None:
    chunks = []
    build_stream = ctx.get_stream_from_pool()
    chunk_streams = set()
    while (msg := await ch_in.recv(ctx)) is not None:
        chunk = TableChunk.from_message(msg)
        chunks.append(chunk)
        chunk_streams.add(chunk.stream)
    join_streams(list(chunk_streams), build_stream)
    table = plc.concatenate.concatenate(
        [chunk.table_view() for chunk in chunks], build_stream
    )
    for s in chunk_streams:
        join_streams([build_stream], s)
    await ch_out.send(
        ctx,
        Message(
            0, TableChunk.from_pylibcudf_table(table, build_stream, exclusive_view=True)
        ),
    )
    await ch_out.drain(ctx)


@define_py_node()
async def sort_by_and_round(
    ctx: Context, ch_in: Channel[TableChunk], ch_out: Channel[TableChunk]
) -> None:
    msg = await ch_in.recv(ctx)
    if msg is None:
        raise RuntimeError("Expecting a chunk in sort by")
    if await ch_in.recv(ctx) is not None:
        raise RuntimeError("Only expecting a single chunk")
    chunk = TableChunk.from_message(msg)
    name, date, revenue = chunk.table_view().columns()
    stream = chunk.stream
    revenue = plc.round.round(revenue, 2, plc.round.RoundingMethod.HALF_EVEN, stream)
    await ch_out.send(
        ctx,
        Message(
            0,
            TableChunk.from_pylibcudf_table(
                plc.sorting.sort_by_key(
                    plc.Table([name, date, revenue]),
                    plc.Table([name, date]),
                    [plc.types.Order.ASCENDING, plc.types.Order.DESCENDING],
                    [plc.types.NullOrder.BEFORE, plc.types.NullOrder.BEFORE],
                    stream,
                ),
                stream,
                exclusive_view=True,
            ),
        ),
    )
    await ch_out.drain(ctx)


@define_py_node()
async def write_parquet(
    ctx: Context, ch_in: Channel[TableChunk], filename: Path
) -> None:
    msg = await ch_in.recv(ctx)
    if msg is None:
        raise RuntimeError("Expecting a chunk in write_parquet")
    if await ch_in.recv(ctx) is not None:
        raise RuntimeError("Only expecting a single chunk in write_parquet")
    chunk = TableChunk.from_message(msg)
    sink = plc.io.SinkInfo([filename])
    builder = plc.io.parquet.ParquetWriterOptions.builder(sink, chunk.table_view())
    metadata = plc.io.types.TableInputMetadata(chunk.table_view())
    metadata.column_metadata[0].set_name("nation")
    metadata.column_metadata[1].set_name("o_year")
    metadata.column_metadata[2].set_name("sum_profit")
    options = builder.metadata(metadata).build()
    plc.io.parquet.write_parquet(options, chunk.stream)


def get_files(base: str, suffix: str) -> list[str]:
    path = Path(base)
    if path.is_dir():
        files = sorted(path.glob(f"*.{suffix}"))
        if len(files) == 0:
            raise RuntimeError(f"No parquet files found in {path}")
        return [str(f) for f in files]
    else:
        path = path.with_suffix(f".{suffix}")
        if not path.exists():
            raise RuntimeError(f"File {path} does not exist")
        return [str(path)]


def q(
    ctx: Context,
    num_rows_per_chunk: int,
    num_producers_per_read: int,
    output: str,
    parquet_suffix: str,
    lineitem: str,
    nation: str,
    orders: str,
    part: str,
    partsupp: str,
    supplier: str,
) -> list[CppNode | PyNode]:
    lineitem_files = get_files(lineitem, parquet_suffix)
    part_files = get_files(part, parquet_suffix)
    partsupp_files = get_files(partsupp, parquet_suffix)
    supplier_files = get_files(supplier, parquet_suffix)
    orders_files = get_files(orders, parquet_suffix)
    nation_files = get_files(nation, parquet_suffix)
    nodes: list[CppNode | PyNode] = []
    lineitem_ch = Channel[TableChunk]()
    part_ch = Channel[TableChunk]()
    filtered_part = Channel[TableChunk]()
    partsupp_ch = Channel[TableChunk]()
    supplier_ch = Channel[TableChunk]()
    orders_ch = Channel[TableChunk]()
    nation_ch = Channel[TableChunk]()
    part_x_partsupp = Channel[TableChunk]()
    supplier_x_part_x_partsupp = Channel[TableChunk]()
    supplier_x_part_x_partsupp_x_lineitem = Channel[TableChunk]()
    supplier_x_part_x_partsupp_x_lineitem_x_orders = Channel[TableChunk]()
    all_joined = Channel[TableChunk]()
    groupby_input = Channel[TableChunk]()
    nodes.append(
        read_part(ctx, part_files, num_producers_per_read, num_rows_per_chunk, part_ch)
    )
    nodes.append(
        read_partsupp(
            ctx, partsupp_files, num_producers_per_read, num_rows_per_chunk, partsupp_ch
        )
    )
    nodes.append(
        read_supplier(
            ctx, supplier_files, num_producers_per_read, num_rows_per_chunk, supplier_ch
        )
    )
    nodes.append(
        read_lineitem(
            ctx, lineitem_files, num_producers_per_read, num_rows_per_chunk, lineitem_ch
        )
    )
    nodes.append(
        read_orders(
            ctx, orders_files, num_producers_per_read, num_rows_per_chunk, orders_ch
        )
    )
    # Nation is tiny so only launch a single producer
    nodes.append(read_nation(ctx, nation_files, num_rows_per_chunk, nation_ch))
    nodes.append(filter_part(ctx, part_ch, filtered_part))
    nodes.append(
        broadcast_join(
            ctx, filtered_part, partsupp_ch, part_x_partsupp, [0], [0], keep_keys=True
        )
    )
    nodes.append(
        broadcast_join(
            ctx,
            supplier_ch,
            part_x_partsupp,
            supplier_x_part_x_partsupp,
            [1],
            [1],
            keep_keys=True,
        )
    )
    nodes.append(
        broadcast_join(
            ctx,
            supplier_x_part_x_partsupp,
            lineitem_ch,
            supplier_x_part_x_partsupp_x_lineitem,
            [2, 1],
            [3, 5],
            keep_keys=False,
        )
    )
    nodes.append(
        broadcast_join(
            ctx,
            supplier_x_part_x_partsupp_x_lineitem,
            orders_ch,
            supplier_x_part_x_partsupp_x_lineitem_x_orders,
            [4],
            [1],
            keep_keys=False,
        )
    )
    nodes.append(
        broadcast_join(
            ctx,
            nation_ch,
            supplier_x_part_x_partsupp_x_lineitem_x_orders,
            all_joined,
            [1],
            [0],
            keep_keys=False,
        )
    )
    nodes.append(select_columns(ctx, all_joined, groupby_input))
    groupby_output = Channel[TableChunk]()
    nodes.append(chunkwise_groupby_agg(ctx, groupby_input, groupby_output))
    concat_output = Channel[TableChunk]()
    nodes.append(concatenate(ctx, groupby_output, concat_output))
    final_grouped = Channel[TableChunk]()
    nodes.append(chunkwise_groupby_agg(ctx, concat_output, final_grouped))
    sorted = Channel[TableChunk]()
    nodes.append(sort_by_and_round(ctx, final_grouped, sorted))
    nodes.append(write_parquet(ctx, sorted, Path(output)))
    return nodes


@click.command()
@click.option(
    "--num-iterations", default=2, help="Number of iterations of the query to run"
)
@click.option("--output", default="result.pq", help="Output result file")
@click.option(
    "--num-rows-per-chunk",
    default=50_000_000,
    help="Number of rows read in a single chunk from input tables",
)
@click.option(
    "--num-producers-per-read",
    default=4,
    help="Number of producer tasks for each parquet read",
)
@click.option(
    "--num-streaming-threads",
    default=8,
    help="Number of threads C++ executor should use",
)
@click.option(
    "--num-py-streaming-threads",
    default=1,
    help="Number of threads Python executor should use",
)
@click.option(
    "--parquet-suffix", default="parquet", help="Suffix to append to find parquet files"
)
@click.option(
    "--lineitem",
    default="lineitem",
    help="Name of file (with suffix appended) or name of directory containing lineitem files",
)
@click.option(
    "--nation",
    default="nation",
    help="Name of file (with suffix appended) or name of directory containing nation files",
)
@click.option(
    "--orders",
    default="orders",
    help="Name of file (with suffix appended) or name of directory containing orders files",
)
@click.option(
    "--part",
    default="part",
    help="Name of file (with suffix appended) or name of directory containing part files",
)
@click.option(
    "--partsupp",
    default="partsupp",
    help="Name of file (with suffix appended) or name of directory containing partsupp files",
)
@click.option(
    "--supplier",
    default="supplier",
    help="Name of file (with suffix appended) or name of directory containing supplier files",
)
def main(
    num_iterations: int,
    output: str,
    num_rows_per_chunk: int,
    num_producers_per_read: int,
    num_streaming_threads: int,
    num_py_streaming_threads: int,
    parquet_suffix: str,
    lineitem: str,
    nation: str,
    orders: str,
    part: str,
    partsupp: str,
    supplier: str,
) -> None:
    py_exec = ThreadPoolExecutor(max_workers=num_py_streaming_threads)
    ctx = get_streaming_context(num_streaming_threads)
    for i in range(num_iterations):
        start = time.perf_counter()
        nodes = q(
            ctx,
            num_rows_per_chunk,
            num_producers_per_read,
            output,
            parquet_suffix,
            lineitem,
            nation,
            orders,
            part,
            partsupp,
            supplier,
        )
        end = time.perf_counter()
        print(f"Iteration {i}: Pipeline construction {end - start:.4g}s")
        with nvtx.annotate(message="Q9 iteration", color="blue", domain="rapidsmpf"):
            start = time.perf_counter()
            run_streaming_pipeline(nodes=nodes, py_executor=py_exec)
            end = time.perf_counter()
        print(f"Iteration {i}: Pipeline execution {end - start:.4g}s")


if __name__ == "__main__":
    main()
