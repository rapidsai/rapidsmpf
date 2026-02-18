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
> where online queries are run on continuously arriving data.


## Concepts

The abstract framework we use to describe task graphs is broadly that of
Hoare's [Communicating Sequential
Processes](https://en.wikipedia.org/wiki/Communicating_sequential_processes).
Actors in the graph are long-lived coroutines that read from zero-or-more
channels and write to zero-or-more channels. In this sense, the programming
model is relatively close to that of
[actors](https://en.wikipedia.org/wiki/Actor_model).

The communication channels are bounded capacity, multi-producer
multi-consumer queues. An actor processing data from an input channel pulls
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

A typical approach is to define one actor in the graph for each physical
operation in the query plan. Parallelism is obtained by using a
multi-threaded executor to handle the concurrent tasks that thus result.

For use with data processing engines, we provide a number of utility tasks
that layer a streaming (out of core) execution model over the
GPU-accelerated [libcudf](https://docs.rapids.ai/api/libcudf/stable/)
library.

```
+------+     +--------+     +--------+     +------+
| Scan | --> | Select | --> | Filter | --> | Sink |
+------+     +--------+     +--------+     +------+
```
*A typical rapidsmpf network of actors*

 Once constructed, the network of actors and their connecting channels remains in place for the duration of the workflow. Each actor continuously awaits new data, activating as soon as inputs are ready and forwarding results downstream via the channels to the next actor(s) in the graph.


## Definitions

```{glossary}
Network
  A graph of actors and edges. Actors are the relational operators on data and edges are the channels connecting the next operation in the workflow.

Context
  Provides access to resources necessary for executing actors:
  - Communicators (UCXX or MPI)
  - Thread pool executor
  - CUDA Memory (RMM)
  - rapidsmpf Buffer Resource (spillable)

Buffer
  Raw memory buffers typically shared pointers from tabular data provided by cuDF.

  - Buffers are created most commonly during scan (read_parquet) operations but can also be created during joins and aggregations. When operating on multiple buffers either a new stream is created for the new buffer or re-use of an existing stream is attached the newly created buffer.
  - Buffers have an attached CUDA Stream maintained for the lifetime of the buffer.

Message
  [Type-erased](https://en.wikipedia.org/wiki/Type_erasure) container for data payloads (shared memory pointers) including: cudf tables, buffers, and rapidsmpf internal data structures like packed data.

  - Messages also contain metadata: a sequence number.
  - Sequences _do not_ guarantee that chunks arrive in order but they do provide the order in which the data was created.

Actor
  Coroutine-based asynchronous relational operator: read, filter, select, join.

  - Actors read from zero-or-more channels and write to zero-or-more channels.
  - Multiple Actors can be executed concurrently.
  - Actors can communicate directly using "streaming" collective operations such as shuffles and joins (see [Streaming collective operations](./shuffle-architecture.md#streaming-collective-operations)).

Channel
  An asynchronous messaging queue used for communicating Messages between Actors.
  - Provides backpressure to the network prevent over consumption of memory
  - Can be throttled to prevent over production of buffers which can useful when writing producer actors that otherwise do not depend on an input channel.
  - Sending suspends when channel is "full".
  - Does not copy (de-)serialize, (un-)spill data

```
