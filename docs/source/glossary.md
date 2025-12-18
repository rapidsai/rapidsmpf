# Glossary

This glossary defines key concepts and terminology used throughout rapidsmpf.

```{glossary}
AllGather
  A collective operation that gathers data from all ranks and distributes the combined result to every rank. Each rank contributes its local data, and after the operation completes, all ranks have a copy of the concatenated data from all participants.

Buffer
  A raw memory allocation that can reside in device (GPU), pinned host, or regular host memory. Buffers have an attached CUDA stream maintained for the lifetime of the buffer. They are typically created during scan (read) operations or when new data is produced by joins and aggregations.

BufferResource
  A class that manages memory allocation and transfers between different memory types (device, pinned host, and host). All memory operations in rapidsmpf, such as those performed by the Shuffler, rely on a BufferResource for memory management. It handles memory reservations, spilling, and provides access to CUDA stream pools.

Channel
  An asynchronous messaging queue used for communicating {term}`Message`s between {term}`Node`s in a streaming network. Channels provide backpressure to prevent memory overconsumption by suspending senders when full. They do not copy, serialize, or spill data - they simply pass references between nodes.

Collective Operation
  A communication pattern that involves coordination across multiple ranks. Examples include {term}`Shuffler` (redistributing data by partition), {term}`AllGather` (gathering data from all ranks), and {term}`Fanout` (broadcasting to multiple channels). These operations are handled internally within nodes while maintaining CSP semantics.

Communicator
  An abstract interface for sending and receiving messages between ranks (processes/GPUs). Communicators support asynchronous operations, GPU data transfers, and custom logging. Implementations include UCXX (for UCX-based communication) and MPI backends.

Context
  The execution environment for {term}`Node`s in a streaming pipeline. A Context provides access to essential resources including:
  - A {term}`Communicator` for inter-rank communication
  - A {term}`BufferResource` for memory management
  - A {term}`ProgressThread` for background operations
  - A coroutine thread pool executor
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
  A directed graph of {term}`Node`s connected by {term}`Channel`s representing a streaming data processing pipeline. Nodes are relational operators on data, and channels are the edges connecting operations in the workflow. The network remains in place for the duration of a workflow, with nodes continuously processing data as it flows through.

Node
  A coroutine-based asynchronous operator in a streaming pipeline. Nodes read from zero or more input {term}`Channel`s, perform computation, and write to zero or more output channels. They can be local (operating on data within a single rank) or collective (coordinating across multiple ranks). Multiple nodes execute concurrently via a thread pool executor.

Options
  A configuration container that stores key-value pairs controlling rapidsmpf behavior. Options can be populated from environment variables (prefixed with `RAPIDSMPF_`) or set programmatically. Common options include logging verbosity, memory limits, and integration-specific settings.

PackedData
  Serialized (packed) data ready for transfer between ranks or for spilling to host memory. PackedData contains both metadata and the actual data buffers in a format that can be efficiently transmitted and later unpacked back into structured data like cuDF tables.

Partition
  A logical division of data assigned to a particular rank during shuffle operations. Data is partitioned using hash-based or custom partitioning schemes, with each partition identified by a unique partition ID (PartID). The {term}`Shuffler` redistributes partitions so that each rank receives all data belonging to its assigned partitions.

Payload
  A protocol that message payloads must implement to be sent through {term}`Channel`s. The protocol defines how to construct a payload from a {term}`Message` and how to insert a payload back into a message, enabling type-safe communication between nodes.

ProgressThread
  A background thread that executes registered progress functions in a loop. Used by the {term}`Shuffler` and other components to make continuous progress on asynchronous operations without blocking the main execution. Functions can be dynamically added and removed, and the thread can be paused and resumed.

Rank
  A unique integer identifier for a process in a distributed system, ranging from 0 to nranks-1. Each rank typically corresponds to one GPU. The rank is used to determine which partitions a process owns and to route messages between processes.

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
