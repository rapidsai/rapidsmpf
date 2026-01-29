# Glossary

This glossary defines key concepts and terminology used throughout rapidsmpf.

```{glossary}
AllGather
  A collective operation that gathers data from all ranks and distributes the combined result to every rank. Each rank contributes its local data, and after the operation completes, all ranks have a copy of the concatenated data from all participants.

Buffer
  A raw memory allocation that can reside in device (GPU), pinned host, or regular host memory. Buffers have an attached CUDA stream maintained for the lifetime of the buffer; all operations on the buffer are stream-ordered, including when the underlying storage is host memory. They are created through a {term}`BufferResource`.

BufferResource
  A class that manages memory allocation and transfers between different memory types (device, pinned host, and host). All memory operations in rapidsmpf, such as those performed by the Shuffler, rely on a BufferResource for memory management. It handles memory reservations, spilling, and provides access to CUDA stream pools.

Channel
  An asynchronous messaging queue used for communicating {term}`Message`s between {term}`Node`s within a single local network instance on a {term}`Rank`. Channels provide backpressure to prevent excessive memory consumption by suspending senders when full. They do not copy, serialize, spill, or transmit data across ranks; instead, they simply pass references between local nodes. Any inter-rank communication is handled explicitly by nodes via a {term}`Communicator`, outside the channel abstraction.

Collective Operation
  A communication pattern that involves coordination across multiple {term}`Rank`s and is performed within a {term}`Node`. Collective operations use a {term}`Communicator` to exchange data between ranks, while remaining fully encapsulated within the node's execution. From the network's perspective, a collective operation is part of a local node's computation and does not alter the network topology or channel semantics. Examples include {term}`Shuffler` (redistributing data by partition) and {term}`AllGather` (gathering data from all ranks). Operations such as {term}`Fanout`, which broadcast to multiple channels within the local network, are **not** collective operations because they do not involve inter-rank communication.

Communicator
  An abstract interface for sending and receiving messages between {term}`Rank`s (processes/GPUs). Communicators support asynchronous operations, GPU data transfers, and custom logging. rapidsmpf includes UCXX (for UCX-based communication) and MPI-based communicators. A single-process communicator can be used for testing.

Context
  The execution environment for {term}`Node`s in a streaming pipeline. A Context provides access to essential resources including:
  - A {term}`Communicator` for inter-rank communication
  - A {term}`BufferResource` for memory management
  - A {term}`ProgressThread` for background operations
  - A thread pool for executing coroutines
  - Configuration {term}`Options`
  - {term}`Statistics` for performance tracking

Fanout
  A streaming operation that broadcasts messages from a single input {term}`Channel` to multiple output channels. Supports both bounded and unbounded policies for controlling message delivery.

MemoryReservation
  A token representing a promise of future memory allocation from a {term}`BufferResource`. Reservations must be obtained before allocating buffers, enabling the system to track memory usage and perform spilling when necessary. Reservations specify the memory type (device, pinned host, or host) and size.

MemoryType
  An enumeration specifying the location of memory:
  - `DEVICE`: GPU memory
  - `PINNED_HOST`: Page-locked host memory for efficient GPU transfers
  - `HOST`: Regular system memory

Message
  A type-erased container for data payloads passed through {term}`Channel`s. Messages wrap arbitrary payload types (such as cuDF tables or buffers) along with metadata including a sequence number for ordering. Messages support deep-copy operations and can be spilled to different memory types when memory pressure occurs.

Network
  A directed graph of {term}`Node`s connected by {term}`Channel`s representing a streaming data processing pipeline local to a single {term}`Rank`. From the network's point of view, all nodes and channels are local, even if individual nodes internally perform inter-rank communication. The network topology is identical on every participating rank, which ensures consistent execution semantics across the distributed system. The network remains in place for the duration of a workflow, with nodes continuously processing data as data flows through.

Node
  A coroutine-based asynchronous operator in a streaming pipeline. Nodes receive from zero or more input {term}`Channel`s, perform computation, and send to zero or more output channels. From the network's perspective, nodes are always local operators. A node may internally use a {term}`Communicator` to perform inter-rank communication, but this coordination is fully encapsulated within the node and is not visible to the surrounding network or channels. Multiple nodes execute concurrently via a thread pool executor.

Options
  A configuration container that stores key-value pairs controlling rapidsmpf behavior. Options can be populated from environment variables (prefixed with `RAPIDSMPF_`) or set programmatically. Common options include logging verbosity, memory limits, and integration-specific settings.

PackedData
  Serialized (packed) data ready for transfer between ranks or for spilling to host memory. PackedData contains both metadata and the actual data buffers in a format that can be efficiently transmitted and later unpacked back into structured data like cuDF tables.

Partition
  A logical division of data assigned to a particular rank during shuffle operations. Data is partitioned using hash-based or custom partitioning schemes, with each partition identified by a unique partition ID. The {term}`Shuffler` redistributes partitions so that each rank receives all data belonging to its assigned partitions.

ProgressThread
  A background thread that executes registered progress functions in a loop. Used by the {term}`Shuffler` and other components to make continuous progress on asynchronous operations (including spilling and unspilling) without blocking the main execution. Functions can be dynamically added and removed, and the thread can be paused and resumed.

Rank
  A unique integer identifier for a process in a distributed system, ranging from 0 to nranks-1. Each rank typically corresponds to one GPU. The rank is used to determine how to distribute work among processes and how to route messages between processes.

Shuffler
  A service for redistributing partitioned data across ranks. The Shuffler accepts packed data chunks, routes them to the appropriate destination ranks based on partition ownership, and allows extraction of completed partitions. It supports asynchronous operation with pipelining of insertions and extractions, and can spill data to host memory under memory pressure.

SpillableMessages
  A collection that manages {term}`Message`s that can be spilled to different memory types (typically from device to host memory) when GPU memory is scarce. Messages are inserted with a unique ID and can be extracted or spilled on demand, enabling out-of-core processing of data larger than GPU memory.

SpillManager
  A component that coordinates memory spilling across different parts of the system. The SpillManager maintains a registry of spill functions with priorities, and when memory pressure occurs, it invokes these functions to free up memory by moving data from device memory to host memory or storage.

Spilling
  The process of moving data from GPU (device) memory to host memory or storage when GPU memory is scarce. Spilling enables out-of-core processing where the working set exceeds available GPU memory. Data is later "unspilled" (moved back to GPU memory) when needed for computation.

Statistics
  A class for collecting and reporting performance metrics during rapidsmpf operations. Statistics tracks various counters (bytes transferred, operations performed, timing information) and can optionally profile memory allocations. Statistics can be enabled/disabled and provides a formatted report of collected metrics.
```
