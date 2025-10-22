import asyncio
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
from rapidsmpf.utils.cudf import cudf_to_pylibcudf_table

# Configuration for backpressure control
MAX_IO_THREADS = 10  # Limit concurrent I/O operations
io_throttle = asyncio.Semaphore(MAX_IO_THREADS)

# Setup streams for GPU operations
STREAMS = [Stream() for _ in range(4)]

def get_stream(chunk_id: int) -> Stream:
    return STREAMS[chunk_id % len(STREAMS)]

# Initialize context
options = Options(get_environment_variables())
ctx = Context(
    comm=new_communicator(options),
    br=BufferResource(RmmResourceAdaptor(rmm.mr.get_current_device_resource())),
    options=options,
)
py_executor = ThreadPoolExecutor(max_workers=4)


def create_chunk(chunk_id: int, rows: int = 1000):
    """Create a cuDF table chunk with synthetic data."""
    df = cudf.DataFrame({
        "id": cp.arange(rows, dtype=cp.int32),
        "value": cp.random.rand(rows, dtype=cp.float32),
        "chunk_id": cp.full(rows, chunk_id, dtype=cp.int32)
    })
    table = cudf_to_pylibcudf_table(df)
    return table


@define_py_node()
async def producer(ctx: Context, ch_out: Channel, num_chunks: int = 10):
    """
    Producer node that generates data chunks.
    Uses io_throttle semaphore to control backpressure during chunk creation.
    """
    print(f"[Producer] Starting to produce {num_chunks} chunks")
    
    for i in range(num_chunks):
        # Acquire semaphore to limit concurrent I/O operations
        async with io_throttle:
            print(f"[Producer] Creating chunk {i} (semaphore acquired)")
            
            stream = get_stream(i)
            loop = asyncio.get_running_loop()
            
            # Simulate I/O-bound work (e.g., reading from disk/network)
            await asyncio.sleep(0.1)  # Simulate I/O delay
            
            # Create chunk in executor to avoid blocking
            table = await loop.run_in_executor(py_executor, create_chunk, i)
            
            chunk = TableChunk.from_pylibcudf_table(
                sequence_number=i,
                table=table,
                stream=stream,
                exclusive_view=True
            )
            
            await ch_out.send(ctx, Message(chunk))
            print(f"[Producer] Sent chunk {i} (semaphore released)")
    
    await ch_out.drain(ctx)
    print("[Producer] Finished producing all chunks")


@define_py_node()
async def transform(ctx: Context, ch_in: Channel, ch_out: Channel):
    """
    Transform node that processes chunks.
    Uses io_throttle semaphore to control backpressure during processing.
    """
    print("[Transform] Starting transformation")
    processed_count = 0
    
    while (msg := await ch_in.recv(ctx)) is not None:
        chunk = TableChunk.from_message(msg)
        seq = chunk.sequence_number
        
        # Acquire semaphore to limit concurrent processing operations
        async with io_throttle:
            print(f"[Transform] Processing chunk {seq} (semaphore acquired)")
            
            # Simulate compute-intensive transformation
            await asyncio.sleep(0.2)  # Simulate processing time
            
            loop = asyncio.get_running_loop()
            
            # Perform transformation in executor
            def apply_transform(chunk):
                # In a real scenario, this would modify the table
                # Here we just pass it through as an example
                return chunk
            
            transformed_chunk = await loop.run_in_executor(
                py_executor, 
                apply_transform, 
                chunk
            )
            
            await ch_out.send(ctx, Message(transformed_chunk))
            processed_count += 1
            print(f"[Transform] Processed chunk {seq} (total: {processed_count}, semaphore released)")
    
    await ch_out.drain(ctx)
    print(f"[Transform] Finished transforming {processed_count} chunks")


@define_py_node()
async def consumer(ctx: Context, ch_in: Channel):
    """
    Consumer node that receives and processes final chunks.
    Uses io_throttle semaphore to control backpressure during consumption.
    """
    print("[Consumer] Starting consumption")
    received_order = []
    
    while (msg := await ch_in.recv(ctx)) is not None:
        chunk = TableChunk.from_message(msg)
        seq = chunk.sequence_number
        
        # Acquire semaphore to limit concurrent I/O operations (e.g., writing to disk)
        async with io_throttle:
            print(f"[Consumer] Consuming chunk {seq} (semaphore acquired)")
            
            # Simulate I/O-bound work (e.g., writing to disk/network)
            await asyncio.sleep(0.15)  # Simulate I/O delay
            
            received_order.append(seq)
            print(f"[Consumer] Consumed chunk {seq} (semaphore released)")
    
    print("\n" + "="*60)
    print(f"[Consumer] Finished consuming all chunks")
    print(f"[Consumer] Received order: {received_order}")
    print(f"[Consumer] Total chunks: {len(received_order)}")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Create channels connecting the nodes
    ch_producer_to_transform = Channel()
    ch_transform_to_consumer = Channel()
    
    print("\n" + "="*60)
    print("Running streaming pipeline with backpressure control")
    print(f"MAX_IO_THREADS = {MAX_IO_THREADS}")
    print("="*60 + "\n")
    
    # Run the pipeline
    run_streaming_pipeline(
        nodes=(
            producer(ctx, ch_out=ch_producer_to_transform, num_chunks=10),
            transform(ctx, ch_in=ch_producer_to_transform, ch_out=ch_transform_to_consumer),
            consumer(ctx, ch_in=ch_transform_to_consumer)
        ),
        py_executor=py_executor
    )
    
    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print("="*60 + "\n")

