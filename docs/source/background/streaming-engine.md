# Streaming execution

In addition to communications primitives, rapidsmpf provides building
blocks for constructing and executing task graphs such as might be used in
a streaming data processing engine.  These communications primitives do not 
require use of the task execution framework, nor does use of the execution 
framework necessarily require using rapidsmpf communication primitives.

The goal is to enable pipelined "out of core" execution for tabular data
processing tasks where one is processing data that does not all fit in GPU
memory at once.

> The term _streaming_ is somewhat overloaded. In `rapidsmpf`, we mean
> execution on fixed size input data that we process piece by piece because
> it does not all fit in GPU memory at once, or we want to leverage multi-GPU
> parallelism and task launch pipelining.
> 
> This contrasts with "streaming analytics" or "event stream processing"
> where online queries are run on continously arriving data.


## Concepts

The abstract framework we use to describe task graphs is broadly that of
Hoare's [Communicating Sequential
Processes](https://en.wikipedia.org/wiki/Communicating_sequential_processes).
Nodes (tasks) in the graph are long-lived that read from zero-or-more
channels and write to zero-or-more channels. In this sense, the programming
model is relatively close to that of
[actors](https://en.wikipedia.org/wiki/Actor_model).

The communication channels are bounded capacity, multi-producer
multi-consumer queues. A node processing data from an input channel pulls
data as necessary until the channel is empty, and can optionally signal
that it needs no more data (thus shutting the producer down).

Communication between tasks in the same process occurs through channels. In
contrast communication between processes uses the lower-level rapidsmpf
communication primitives. In this way, achieving forward progress of the
task graph is a local property, as long as the logically collective
semantics of individual tasks are obeyed internally.

The recommended usage to target multiple GPUs is to have one process per
GPU, tied together by a rapidsmpf communicator.


## Building task networks from query plans

The task specification is designed to be lowered to from some higher-level
application specific intermediate representation, though one can write it
by hand.  For example, one can convert logical plans from query engines such as 
Polars, DuckDB, etc to a physical plan to be executed by rapidsmpf.

A typical approach is to define one node in the graph for each physical
operation in the query plan. Parallelism is obtained by using a
multi-threaded executor to handle the concurrent tasks that thus result.

For use with data processing engines, we provide a number of utility tasks
that layer a streaming (out of core) execution model over the
GPU-accelerated [libcudf](https://docs.rapids.ai/api/libcudf/stable/)
library.


```
     Source
        |
        |
     Node 1
     /    \
    /      \
 Node 2  node 3
     \    /
      \  /
    Accumulator
        |
        |
      Sink
```


*A typical rapidsmpf network of nodes*

 Once constructed, the network of "nodes" and their connecting channels remains in place for the duration of the workflow. Each node continuously awaits new data, activating as soon as inputs are ready and forwarding results downstream via the channels to the next node(s) in the graph.


## Definitions
- **Network**: A graph of nodes and edges.  `Nodes` are the relational operators on data and edges are the `channels` connecting the _next_ operation in the workflow

- **Context**: Context provides access to resources necessary for executing nodes:
  - Communicators (UCXX or MPI)
  - Thread pool executor
  - CUDA Memory (RMM) 
  - rapidsmpf Buffer Resource (spillable)

- **Buffer** : Raw Memory buffers typically shared pointers from tabular data provided by cuDF
  - Buffers are created most commonly during scan (read_parquet) operations but can also be created during joins and aggregations.  When operating on mulitple buffers either a new stream is created for the new buffer or re-use of an existing stream is attached the newly created buffer
  - Buffers have an attached CUDA Stream maintained for the lifetime of the buffer. 
  
- **Messages**:[Type-erased](https://en.wikipedia.org/wiki/Type_erasure) container for data payloads (shared memory pointers) including: cudf tables, buffers, and rapidsmpf internal data structures like packed data
  - Messages also contain metadata: a sequence number
  - Sequences _do not_ guarantee that chunks arrive in order but they do provide the order in which the data was created

- **Nodes**: Coroutine-based asynchronous relational operator: read, filter, select, join.  
  - Nodes read from zero-or-more channels and write to zero-or-more channels
  - Multiple Nodes can be executed concurrently
  - Nodes can communicate directly using "streaming" collective operations such as shuffles and joins (see [Streaming collective operations](./shuffle-architecture.md#streaming-collective-operations)).

- **Channels**: An asynchronous messaging queue used for communicating Messages between Nodes.
  - Can be throttled to prevent over production of buffers which can useful when writing producer nodes that otherwise do not depend on an input channel.
  - Sending suspends when channel is "full"