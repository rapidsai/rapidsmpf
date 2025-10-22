# Streaming Engine

## What is it ?

RapidsMPF is a backend streaming data processing engine using a [CSP](https://en.wikipedia.org/wiki/Communicating_sequential_processes)-style streaming network written in C++ with additional hooks for Python.  RapidsMPF is designed to integrate with frontend query processing libraries (e.g., Polars) where logical plans are converted and executed by RapidsMPF.  In Multi-GPU deployments, plans are replicated to all workers, where each worker is operating on its local data partition.  RapidsMPF has also implemented out-of-core streaming shuffles for both broadcast and hash joins.  See here for more info on [streaming shuffles](./shuffle-architecture.md)


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
*Execution on each worker is structured as a network of CSP-processes operating in a streaming fashion.*

  As data becomes available to a CSP-process (`Node`), it begins computation and forwards results downstream in the graph.  These "nodes" execute operations as asynchronous coroutines on multipe [CUDA Streams](https://developer.nvidia.com/blog/gpu-pro-tip-cuda-7-streams-simplify-concurrency/) -- overlapping of I/O, computation, and communication.  

```
Time --->

scan:     |====|====|====|====|
                \    \    \    \
compute:   .----|====|====|====|====|
                \    \    \    \
sink:      ......|====|====|====|====|

Legend:
|====| : Active period of the task
\    : Data passed downstream as it becomes available
.    : Waiting or not yet started
```
*Overlapping execution of of scan, compute, sink*


`scan`, `compute`, and `sink` all operate in a pipelined fashion as data is pushed through the execution graph 

RapidsMPF also natively supports out-of-core processing where buffers can move seamlessly between device and host and be communicated as either host or device buffers when collective operations: shuffles, groupby-aggregrations, etc are used.


## RapidsMPF Streaming Network

RapidsMPF builds a network of workers where each worker starts one process/multiple threads and is pinned to one GPU.  Query plans are transformed into a list of nodes (executing coroutines) where each Node performs a single relational operation on a single data chunk at a time.  Nodes can operate and different chunks of data concurrently by operating data on independent [CUDA streams](https://developer.nvidia.com/blog/gpu-pro-tip-cuda-7-streams-simplify-concurrency/)

Each process has a set of input channels it reads from and output channels it writes to. For example:

- A parquet-read CSP-process has only an output channel — it generates data chunks.
- A filtering CSP-process has both input and output — it reads data, applies a predicate, and emits filtered results.

We execute a workflow in a streaming fashion, where Nodes (coroutines) starting executing as soon as input data becomes available.  Channels "pass" data between Nodes and help reduce OOM errors by limiting the amount of buffers into a channel and creating backpressure. 

Buffers are maintained with a Buffer Manager so that we can move data seamlessly between device and host enabling out-of-core processing and spilling.

```
+------+     +--------+     +--------+     +------+
| Scan | --> | Select | --> | Filter | --> | Sink |
+------+     +--------+     +--------+     +------+
```
*example of a simple RapidsMPF graph*

## Definitions
- **Network**: A graph of nodes and edges.  `Nodes` are the relational operators on data and edges are the `channels` connecting the _next_ operation in the workflow

- **Context**: Context provides access to resources necessary for executing nodes:
  - UCXX communicators
  - Thread pool executor
  - CUDA Memory (RMM) 
  - RapidMPF Buffer Resource (spillable)

- **Buffer** : Raw Memory buffers typically shared pointers from tabular data provided by cuDF
  - Buffers are created mostly commonly during scan (read_parquet) operations but can also be created during joins and aggregations.  When operating on mulitple buffers either a new stream is created for the new buffer or re-use of an existing stream to attach the newly created buffer

  - Buffers have an attached CUDA Stream maintained for the lifetime of the buffer. 
  - Streams are created or used from an existing stream pool
  
- **Messages**:[Type-erased](https://en.wikipedia.org/wiki/Type_erasure) container for data payloads (shared memory pointers) including: cudf tables, buffers, and RapidsMPF internal data structures like packed data
  - Messages also contain metadata like a sequence number
  - Sequences _do not_ guarantee that chunks arrive in order but they do provide the order in which the data was created

- **Nodes**: Coroutine-based asynchronous relational operator: read, filter, select, join.  
  - Multiple Nodes can be executed concurrently
  - Nodes send/recv data on a channel, process it, and send the result to an output channel
  - Nodes can communicate with each other directly like in the cases of: shuffles, joins, etc.

- **Channels**: An asynchronous messaging queue used for forward progressing `messages`.
  - Can be throttled to prevent over production of messages – useful when writing producer nodes that otherwise do not depend on an input channel.
  - Throttling limits the number of concurrent tasks/nodes
Sending suspends when channel is full