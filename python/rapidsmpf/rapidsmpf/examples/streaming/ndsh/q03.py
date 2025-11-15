# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: D103

"""Example by hand implementation of a derivation of TPC-H Q3."""

from __future__ import annotations

import itertools
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
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


def read_customer(
    ctx: Context,
    files: Sequence[str],
    num_producers: int,
    num_rows_per_chunk: int,
    ch: Channel[TableChunk],
) -> CppNode | PyNode:
    columns = [
        "c_mktsegment",  # 0
        "c_custkey",  # 1
    ]
    return read_parquet(
        ctx, ch, num_producers, reader_options(files, columns), num_rows_per_chunk
    )

def read_lineitem(
    ctx: Context,
    files: Sequence[str],
    num_producers: int,
    num_rows_per_chunk: int,
    ch: Channel[TableChunk],
) -> CppNode | PyNode:
    columns = [
        "l_orderkey",  # 0
        "l_shipdate",  # 1
        "l_extendedprice", # 2
        "l_discount", # 3
    ]
    return read_parquet(
        ctx, ch, num_producers, reader_options(files, columns), num_rows_per_chunk
    )

def read_orders(
    ctx: Context,
    files: Sequence[str],
    num_producers: int,
    num_rows_per_chunk: int,
    ch: Channel[TableChunk],
) -> CppNode | PyNode:
    columns = [
        "o_orderkey",  # 0
        "o_orderdate", # 1
        "o_shippriority", # 2
        "o_custkey", # 3
    ]
    return read_parquet(
        ctx, ch, num_producers, reader_options(files, columns), num_rows_per_chunk
    )


# customer.filter(pl.col("c_mktsegment") == var1) ## var1 = "BUILDING"
@define_py_node()
async def filter_customer(
    ctx: Context, ch_in: Channel[TableChunk], ch_out: Channel[TableChunk]
) -> None:
    while (msg := await ch_in.recv(ctx)) is not None:
        chunk = TableChunk.from_message(msg)
        stream = chunk.stream
        table = chunk.table_view()
        target = plc.Scalar.from_py("BUILDING", stream=stream)
        chunk = TableChunk.from_pylibcudf_table(
            plc.stream_compaction.apply_boolean_mask(
                plc.Table(table.columns()[1:]), # no longer need c_mktsegment
                plc.strings.find.contains(table.columns()[0], target, stream), # c_mktsegment is col 0
                stream,
            ),
            stream,
            exclusive_view=True,
        )
        await ch_out.send(ctx, Message(msg.sequence_number, chunk))
    await ch_out.drain(ctx)


# .filter(pl.col("o_orderdate") < var2) ## var2 = date(1995, 3, 15)
@define_py_node()
async def filter_orders(
    ctx: Context, ch_in: Channel[TableChunk], ch_out: Channel[TableChunk]
) -> None:
    var2 = date(1995, 3, 15)
    while (msg := await ch_in.recv(ctx)) is not None:
        chunk = TableChunk.from_message(msg)
        stream = chunk.stream
        table = chunk.table_view()
        target = plc.Scalar.from_py(var2, stream=stream)
        chunk = TableChunk.from_pylibcudf_table(
            plc.stream_compaction.apply_boolean_mask(
                plc.Table(table.columns()[:]), # still need all columns
                plc.binaryop.binary_operation(
                    table.columns()[1], # o_orderdate is col 1
                    target,
                    plc.binaryop.BinaryOperator.LESS,
                    plc.DataType(plc.TypeId.BOOL8),
                    stream),
                stream,
            ),
            stream,
            exclusive_view=True,
        )
        await ch_out.send(ctx, Message(msg.sequence_number, chunk))
    await ch_out.drain(ctx)

# .filter(pl.col("l_shipdate") > var2) ## var2 = date(1995, 3, 15)
@define_py_node()
async def filter_lineitem(
    ctx: Context, ch_in: Channel[TableChunk], ch_out: Channel[TableChunk]
) -> None:
    var2 = date(1995, 3, 15)
    while (msg := await ch_in.recv(ctx)) is not None:
        chunk = TableChunk.from_message(msg)
        stream = chunk.stream
        table = chunk.table_view()
        target = plc.Scalar.from_py(var2, stream=stream)
        chunk = TableChunk.from_pylibcudf_table(
            plc.stream_compaction.apply_boolean_mask(
                plc.Table([table.columns()[i] for i in [0, 2, 3]]), # no longer need l_shipdate
                plc.binaryop.binary_operation(
                    table.columns()[1], # l_shipdate is col 1
                    target,
                    plc.binaryop.BinaryOperator.GREATER,
                    plc.DataType(plc.TypeId.BOOL8),
                    stream),
                stream,
            ),
            stream,
            exclusive_view=True,
        )
        await ch_out.send(ctx, Message(msg.sequence_number, chunk))
    await ch_out.drain(ctx)



@define_py_node()
async def with_columns(
    ctx: Context, ch_in: Channel[TableChunk], ch_out: Channel[TableChunk]
) -> None:
    while (msg := await ch_in.recv(ctx)) is not None:
        chunk = TableChunk.from_message(msg)
        stream = chunk.stream
        columns = chunk.table_view().columns()

        """
        customer_x_orders_x_lineitem is the input to the with_column op
        "c_custkey", # 0 (customers<-orders on o_custkey)
        "o_orderkey", # 1 (orders<-lineitem on o_orderkey)
        "o_orderdate", # 2
        "o_shippriority", # 3
        "l_shipdate", # 4
        "l_extendedprice", # 5
        "l_discount", # 6
        """

        c_custkey, o_orderkey, o_orderdate, o_shippriority, l_extendedprice, l_discount = columns
        revenue_type = l_discount.type()  # float64 since not using decimal
        revenue = plc.transform.transform(
            [l_discount, l_extendedprice],
            """
            static __device__ void calculate_amount(
                double *amount, double discount, double extprice
            ) {
                *amount = extprice * (1 - discount);
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
                    # this is the input to the groupby
                    # only need four columns from this point forward
                    plc.Table([o_orderkey, o_orderdate, o_shippriority, revenue]),
                    stream,
                    exclusive_view=True,
                ),
            ),
        )
    await ch_out.drain(ctx)


@define_py_node()
async def select_columns(
    ctx: Context, ch_in: Channel[TableChunk], ch_out: Channel[TableChunk]
) -> None:
    while (msg := await ch_in.recv(ctx)) is not None:
        chunk = TableChunk.from_message(msg)
        stream = chunk.stream
        columns = chunk.table_view().columns()
        o_orderkey, o_orderdate, o_shippriority, revenue = columns
        await ch_out.send(
            ctx,
            Message(
                msg.sequence_number,
                TableChunk.from_pylibcudf_table(
                    # change the column order
                    plc.Table([o_orderkey, revenue, o_orderdate, o_shippriority]),
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
    chunk_streams = []
    while (msg := await left_ch.recv(ctx)) is not None:
        left_tables.append(TableChunk.from_message(msg))
        chunk_streams.append(left_tables[-1].stream)
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
        chunk_streams.append(chunk.stream)
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
        o_orderkey, o_orderdate, o_shippriority, revenue = chunk.table_view().columns()
        stream = chunk.stream
        grouper = plc.groupby.GroupBy(
            plc.Table([o_orderkey, o_orderdate, o_shippriority]),
            plc.types.NullPolicy.EXCLUDE,
            plc.types.Sorted.NO,
        )
        reqs = [plc.groupby.GroupByRequest(revenue, [plc.aggregation.sum()])]
        (keys, results) = grouper.aggregate(reqs, stream)
        del chunk, o_orderkey, o_orderdate, o_shippriority, revenue
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
    chunk_streams = []
    while (msg := await ch_in.recv(ctx)) is not None:
        chunk = TableChunk.from_message(msg)
        chunks.append(chunk)
        chunk_streams.append(chunk.stream)
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


# .sort(by=["revenue", "o_orderdate"], descending=[True, False])
@define_py_node()
async def sort_by(
    ctx: Context, ch_in: Channel[TableChunk], ch_out: Channel[TableChunk]
) -> None:
    msg = await ch_in.recv(ctx)
    if msg is None:
        raise RuntimeError("Expecting a chunk in sort by")
    if await ch_in.recv(ctx) is not None:
        raise RuntimeError("Only expecting a single chunk")
    chunk = TableChunk.from_message(msg)
    o_orderkey, revenue, o_orderdate, o_shippriority = chunk.table_view().columns()
    stream = chunk.stream
    await ch_out.send(
        ctx,
        Message(
            0,
            TableChunk.from_pylibcudf_table(
                plc.sorting.sort_by_key(
                    plc.Table([o_orderkey, revenue, o_orderdate, o_shippriority]),
                    plc.Table([revenue, o_orderdate]),
                    [plc.types.Order.DESCENDING, plc.types.Order.ASCENDING],
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
async def head(
    ctx: Context, ch_in: Channel[TableChunk], ch_out: Channel[TableChunk]
) -> None:
    while (msg := await ch_in.recv(ctx)) is not None:
        chunk = TableChunk.from_message(msg)
        stream = chunk.stream
        table = chunk.table_view()

        await ch_out.send(
            ctx,
            Message(
                msg.sequence_number,
                TableChunk.from_pylibcudf_table(
                    plc.copying.slice(
                        table,
                        [0, 10],
                        stream,
                    )[0],
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
    metadata.column_metadata[0].set_name("l_orderkey")
    metadata.column_metadata[1].set_name("revenue")
    metadata.column_metadata[2].set_name("o_orderdate")
    metadata.column_metadata[3].set_name("o_shippriority")
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
    customer: str,
    lineitem: str,
    orders: str,
) -> list[CppNode | PyNode]:
    customer_files = get_files(customer, parquet_suffix)
    lineitem_files = get_files(lineitem, parquet_suffix)
    orders_files = get_files(orders, parquet_suffix)

    nodes: list[CppNode | PyNode] = []

    customer_ch = ctx.create_channel()
    lineitem_ch = ctx.create_channel()
    orders_ch = ctx.create_channel()
    
    filtered_customer = ctx.create_channel()
    filtered_orders = ctx.create_channel()
    filtered_lineitem = ctx.create_channel()

    customer_x_orders = ctx.create_channel()
    customer_x_orders_x_lineitem = ctx.create_channel()
    all_joined = ctx.create_channel()

    groupby_input = ctx.create_channel()

    # Read data
    nodes.append(
        read_customer(
            ctx, customer_files, num_producers_per_read, num_rows_per_chunk, customer_ch
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

    # Apply filters
    nodes.append(filter_customer(ctx, customer_ch, filtered_customer))
    nodes.append(filter_lineitem(ctx, lineitem_ch, filtered_lineitem))
    nodes.append(filter_orders(ctx, orders_ch, filtered_orders))
    
    # Join orders into customers
    nodes.append(
        broadcast_join(
            ctx,
            filtered_customer,
            filtered_orders,
            customer_x_orders,
            [0], # c_custkey
            [3], # o_custkey
            keep_keys=True
        )
    )
    # Join lineitem into that combined table
    nodes.append(
        broadcast_join(
            ctx,
            customer_x_orders, # columns 0, 1 from customer, columns 2, 3, 4, 5 from orders
            filtered_lineitem,
            customer_x_orders_x_lineitem,
            [1], # o_orderkey in customer_x_orders 
            [0], # l_orderkey
            keep_keys=True
        )
    )

    # with columns
    nodes.append(with_columns(ctx, customer_x_orders_x_lineitem, groupby_input))

    # groupby aggregation (agg (per chunk) -> concat -> agg (global))
    groupby_output = ctx.create_channel()
    nodes.append(chunkwise_groupby_agg(ctx, groupby_input, groupby_output))
    concat_output = ctx.create_channel()
    nodes.append(concatenate(ctx, groupby_output, concat_output))
    final_grouped = ctx.create_channel()
    nodes.append(chunkwise_groupby_agg(ctx, concat_output, final_grouped))

    # select columns
    select_ch = ctx.create_channel()
    nodes.append(select_columns(ctx, final_grouped, select_ch))

    sorted_ch = ctx.create_channel()
    nodes.append(sort_by(ctx, select_ch, sorted_ch))

    head_ch = ctx.create_channel()
    nodes.append(head(ctx, sorted_ch, head_ch))

    nodes.append(write_parquet(ctx, head_ch, Path(output)))
    return nodes


@click.command()
@click.option(
    "--num-iterations", default=1, help="Number of iterations of the query to run"
)
@click.option("--output", default="result_q03.pq", help="Output result file")
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
    "--customer",
    default="/raid/rapidsmpf/data/tpch/scale-1000/customer",
    help="Name of file (with suffix appended) or name of directory containing nation files",
)
@click.option(
    "--lineitem",
    default="/raid/rapidsmpf/data/tpch/scale-1000/lineitem",
    help="Name of file (with suffix appended) or name of directory containing lineitem files",
)
@click.option(
    "--orders",
    default="/raid/rapidsmpf/data/tpch/scale-1000/orders",
    help="Name of file (with suffix appended) or name of directory containing orders files",
)
def main(
    num_iterations: int,
    output: str,
    num_rows_per_chunk: int,
    num_producers_per_read: int,
    num_streaming_threads: int,
    num_py_streaming_threads: int,
    parquet_suffix: str,
    customer: str,
    lineitem: str,
    orders: str,
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
            customer,
            lineitem,
            orders,
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


# def q3(run_config: RunConfig) -> pl.LazyFrame:
#         """Query 3."""
#         customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
#         lineitem = get_data(run_config.dataset_path, "lineitem", run_config.suffix)
#         orders = get_data(run_config.dataset_path, "orders", run_config.suffix)

#         var1 = "BUILDING"
#         var2 = date(1995, 3, 15)

#         return (
#             customer.filter(pl.col("c_mktsegment") == var1)
#             .join(orders, left_on="c_custkey", right_on="o_custkey")
#             .join(lineitem, left_on="o_orderkey", right_on="l_orderkey")
#             .filter(pl.col("o_orderdate") < var2)
#             .filter(pl.col("l_shipdate") > var2)
#             .with_columns(
#                 (pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).alias(
#                     "revenue"
#                 )
#             )
#             .group_by("o_orderkey", "o_orderdate", "o_shippriority")
#             .agg(pl.sum("revenue"))
#             .select(
#                 pl.col("o_orderkey").alias("l_orderkey"),
#                 "revenue",
#                 "o_orderdate",
#                 "o_shippriority",
#             )
#             .sort(by=["revenue", "o_orderdate"], descending=[True, False])
#             .head(10)
#         )
