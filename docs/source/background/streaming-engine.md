# Streaming execution

In addition to communications primitives, rapidsmpf provides building
blocks for constructing and executing streaming pipelines for use in
data processing engines. These communications primitives do not
require use of the streaming execution framework, nor does use of the execution
framework necessarily require using rapidsmpf communication primitives.

The goal is to enable pipelined "out of core" execution for tabular data
processing workloads where one is processing data that does not all fit in GPU
memory at once.

> The term _streaming_ is somewhat overloaded. In `rapidsmpf`, we mean
> execution on fixed size input data that we process piece by piece because
> it does not all fit in GPU memory at once, or we want to leverage multi-GPU
> parallelism and task launch pipelining.
>
> This contrasts with "streaming analytics" or "event stream processing"
> where online queries are run on continuously arriving data.


## Concepts

The abstract framework we use is broadly that of
Hoare's [Communicating Sequential
Processes](https://en.wikipedia.org/wiki/Communicating_sequential_processes).
{term}`Actor`s in the network are long-lived coroutines that read from zero-or-more
{term}`Channel`s and write to zero-or-more {term}`Channel`s. In this sense, the programming
model is relatively close to that of
[actors](https://en.wikipedia.org/wiki/Actor_model).

The communication channels are bounded capacity, multi-producer
multi-consumer queues. An {term}`Actor` processing data from an input {term}`Channel` pulls
data as necessary until the channel is empty, and can optionally signal
that it needs no more data (thus shutting the producer down).

Communication between actors in the same process occurs through {term}`Channel`s. In
contrast communication between processes uses the lower-level rapidsmpf
communication primitives. In this way, achieving forward progress of the
network is a local property, as long as the logically collective
semantics of individual actors are obeyed internally.

The recommended usage to target multiple GPUs is to have one process per
GPU, tied together by a rapidsmpf {term}`Communicator`.


## Building Actor networks from query plans

Actor networks are designed to be lowered from some higher-level
application specific intermediate representation, though one can write them
by hand.  For example, one can convert logical plans from query engines such as
Polars, DuckDB, etc to a physical plan to be executed by rapidsmpf.

A typical approach is to define one {term}`Actor` in the network for each physical
operation in the query plan. Parallelism is obtained by using a
multi-threaded executor to handle the concurrent actors that thus result.

For use with data processing engines, we provide a number of utility actors
that layer a streaming (out of core) execution model over the
GPU-accelerated [libcudf](https://docs.rapids.ai/api/libcudf/stable/)
library.

```
+------+     +--------+     +--------+     +------+
| Scan | --> | Select | --> | Filter | --> | Sink |
+------+     +--------+     +--------+     +------+
```
*A typical rapidsmpf {term}`Network` of {term}`Actor`s*

 Once constructed, the {term}`Network` of {term}`Actor`s and their connecting {term}`Channel`s remains in place for the duration of the workflow. Each actor continuously awaits new data, activating as soon as inputs are ready and forwarding results downstream via the channels to the next actor(s) in the network.


## Key Concepts

The streaming engine is built around these core concepts (see the {doc}`/glossary` for complete definitions):

- {term}`Network` - A set of {term}`Actor`s connected by {term}`Channel`s
- {term}`Actor` - Coroutine-based asynchronous operators (read, filter, select, join)
- {term}`Channel` - Asynchronous messaging queues with backpressure
- {term}`Message` - Type-erased containers for data payloads
- {term}`Context` - Provides access to resources ({term}`Communicator`, {term}`BufferResource`, etc.)
- {term}`Buffer` - Raw memory allocations with attached CUDA streams
