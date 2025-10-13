import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import cudf
import cupy as cp
import rmm.mr
from rmm.pylibrmm.stream import Stream

from rapidsmpf.buffer.resource import BufferResource
from rapidsmpf.communicator.single import new_communicator
from rapidsmpf.config import Options, get_environment_variables
from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor
from rapidsmpf.streaming.core.channel import Channel, Message
from rapidsmpf.streaming.core.context import Context
from rapidsmpf.streaming.core.node import define_py_node, run_streaming_pipeline
from rapidsmpf.streaming.cudf.table_chunk import TableChunk
from rapidsmpf.utils.cudf import cudf_to_pylibcudf_table, pylibcudf_to_cudf_dataframe

STREAMS = [Stream() for _ in range(4)]

def get_stream(chunk_id: int) -> Stream:
    return STREAMS[chunk_id % len(STREAMS)]


options = Options(get_environment_variables())
ctx = Context(
    comm=new_communicator(options),
    br=BufferResource(RmmResourceAdaptor(rmm.mr.get_current_device_resource())),
    options=options,
)
py_executor = ThreadPoolExecutor(max_workers=4)

def create_chunk():
    df = cudf.DataFrame({"id": cp.arange(10, dtype=cp.int32)})
    table = cudf_to_pylibcudf_table(df)
    return table


@define_py_node()
async def producer_0(ctx: Context, ch_out: Channel):
    """Produces 5 chunks with sequence numbers 0-4, using different streams."""
    for i in range(5):
        stream = get_stream(i)
        stream_id = i % len(STREAMS)
        create_chunk()
        loop = asyncio.get_running_loop()
        table = await loop.run_in_executor(py_executor, create_chunk)
        chunk = TableChunk.from_pylibcudf_table(
            sequence_number=i,
            table=table,
            stream=stream,  # Use different stream for each chunk
            exclusive_view=True
        )

        await ch_out.send(ctx, Message(chunk))

    await ch_out.drain(ctx)

@define_py_node()
async def producer_1(ctx: Context, ch_out: Channel):
    """Produces 5 chunks with sequence numbers 5-9, using different streams."""
    for i in range(5, 10):
        stream = get_stream(i)
        stream_id = i % len(STREAMS)
        create_chunk()
        loop = asyncio.get_running_loop()
        table = await loop.run_in_executor(py_executor, create_chunk)
        chunk = TableChunk.from_pylibcudf_table(
            sequence_number=i,
            table=table,
            stream=stream,  # Use different stream for each chunk
            exclusive_view=True
        )

        await ch_out.send(ctx, Message(chunk))

    await ch_out.drain(ctx)

async def send_chunk(ctx, ch_out, chunk, sleep_time=0):
    await asyncio.sleep(sleep_time)
    msg = Message(chunk)
    await ch_out.send(ctx, msg)

@define_py_node()
async def sleep_transform_concurrent(ctx: Context, ch_left: Channel, ch_right: Channel, ch_out: Channel):
    ch1_done = False
    ch2_done = False

    tasks =[]
    i = 0 

    # concurrently read from two channels
    # tuning delay varies the amount of overlap in ordering at the final consumer
    async def worker(ctx, ch_in, ch_out, delay):
        while (msg := await ch_in.recv(ctx)) is not None:
            if msg is not None:
                chunk = TableChunk.from_message(msg)
                seq = chunk.sequence_number
                await send_chunk(ctx, ch_out, chunk, sleep_time=delay)
    await asyncio.gather(worker(ctx, ch_left, ch_out, .5), worker(ctx, ch_right, ch_out, 0.2))
    await ch_out.drain(ctx)

    # assume ch_left is small and ch_right is large
    # read all of ch_left before right
    while (msg := await ch_left.recv(ctx)) is not None:
        chunk = TableChunk.from_message(msg)
        seq = chunk.sequence_number
        # store then concat all the chunks
    
    while (msg := await ch_right.recv(ctx)) is not None:
        chunk = TableChunk.from_message(msg)
        seq = chunk.sequence_number
        # join chunk with concatatenated table
        # then send immmediately
        await send_chunk(ctx, ch_out, chunk, sleep_time=0)


@define_py_node()
async def sleep_transform_staged(ctx: Context, ch_left: Channel, ch_right: Channel, ch_out: Channel):
    # assume ch_left is small and ch_right is large
    # read all of ch_left before right
    chunks = []
    while (msg := await ch_left.recv(ctx)) is not None:
        chunk = TableChunk.from_message(msg)
        seq = chunk.sequence_number
        # store then concat all the chunks
        chunks.append(chunk)
    
    # Convert the list of TableChunks to cuDF DataFrames -> concat
    dfs = [pylibcudf_to_cudf_dataframe(chunk.table_view()) for chunk in chunks]

    # Concatenate the pylibcudf tables from the TableChunks
    # from pylibcudf.concat import concat as plc_concat
    acc_df = cudf.concat(dfs)
    
    while (msg := await ch_right.recv(ctx)) is not None:
        chunk = TableChunk.from_message(msg)
        seq = chunk.sequence_number
        # join chunk with concatatenated table
        # then send immmediately
    
    msg = Message(chunk)
    await ch_out.send(ctx, msg)
    await ch_out.drain(ctx)

@define_py_node()
async def consumer(ctx: Context, ch_in: Channel):
    """Consumer receives chunks (possibly out of order)."""

    received_order = []

    while (msg := await ch_in.recv(ctx)) is not None:
        chunk = TableChunk.from_message(msg)
        seq = chunk.sequence_number

        received_order.append(seq)
        # print(f"[Consumer] Received chunk {seq}")

    print()
    print(f"[Consumer] Expected order: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]")
    print(f"[Consumer] Actual order:   {received_order}")

# Run pipeline
ch1 = Channel()
ch2 = Channel()
ch3 = Channel()

run_streaming_pipeline(
    nodes=(
        producer_0(ctx, ch_out=ch1),
        producer_1(ctx, ch_out=ch2),
        sleep_transform_staged(ctx, ch_left=ch1, ch_right=ch2, ch_out=ch3),
        consumer(ctx, ch_in=ch3)
    ),
    py_executor=py_executor
)
