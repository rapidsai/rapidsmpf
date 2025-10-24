# Shuffle Architecture

`rapidsmpf` uses a "process-per-GPU" execution model. It can be used both
to run on a single GPU or multiple GPUs. These can either be physically
located within the same multi-GPU machine or spread across multiple
machines. The key requirement is that there exist communication links
between the GPUs.

The core abstraction that encapsulates the set of processes that are
executing collectively is a `Communicator`. This provides unique
identifiers (termed `rank`s) to each process along with message-passing
routes between them. We provide communicator implementations based either
directly on [UCX](https://openucx.org/)/[UCXX](https://github.com/rapidsai/ucxx) or
[MPI](https://www.mpi-forum.org). Message passing handles CPU and GPU data
uniformly, the underlying transport takes care of choosing the appropriate
route.

## "Streaming" collective operations

`rapidsmpf` provides collectives (i.e. communication
primitives) that operate on "streaming" data. As a consequence, a
round of collective communication proceeds in four stages:

1. Participating ranks (defined by the `Communicator`) create a
   collective object.
2. Each rank independently _inserts_ zero-or-more data chunks into the
   collective object.
3. Once a rank has inserted all data chunks, it inserts a _finish marker_.
4. After insertion is finished, a rank can _extract_ data that is the
   result of the collective communication. This may block until data are
   ready.

Collectives over subsets of all ranks in the program are enabled by
creating a `Communicator` object that only contains the desired
participating ranks.

Multiple collective operations can be live at the same time, they are each
distinguished by a `tag`. This `tag` must be consistent across all
participating ranks to line up the messages in the collective.

Notice that we are not responsible for providing the output buffers that a
collective writes into. This is a consequence of the streaming design: to
allocate output buffers of the correct size we would first have to see all
inputs. Instead `rapidsmpf` is responsible for allocation of output buffers
and spilling data from device to host if device memory is at a premium.
However, although `rapidsmpf` allocates outputs it never interprets your
data: it just sends and receives bytes "as-is".

## Shuffles

A key collective operation in large-scale data analytics is a "shuffle"
(a generalised all-to-all). In a shuffle, every participating rank sends
data to every other rank. We will walk through a high-level overview of the
steps in a shuffle using `rapidsmpf` to see how things fit together.

Having created a collective shuffle operation (a `rapidsmpf::Shuffler`), at
a high level, a shuffle operation involves these steps:

1. [user code] Each rank *inserts* **chunks** of data to the Shuffler,
   followed by a finish marker.
2. [rapidsmpf] The Shuffler on that rank processes that chunk by either sending it to
   another rank or keeping it for itself.
3. [user code] Each rank *extracts* chunks of data from each once it's
   ready.

There are more details around how chunks are assigned to output ranks and how memory is
managed. But at a high level, your program is responsible for inserting chunks
somewhere and extracting (the now shuffled) chunks once they've been moved to
the correct rank.

This diagram shows a network of with three ranks in the middle of a Shuffle operation.

![A diagram showing a shuffle.](../_static/rapidsmpf-shuffler-transparent-fs8.png)

As your program inserts chunks of data (see below), each chunk is assigned to
a particular rank. In the diagram above, this is shown by color: each
process (recall a process is uniquely identified by a `(rank,
communicator)` pair) has a particular color (the color of its circle) and each chunk with that color will
be sent to its matching rank. So, for example, all of the green chunks will be
extracted from the green process in the top-left. Note that the number of different
chunk types (colors in this diagram) is typically larger than the number of ranks,
and so each process will be responsible for multiple output chunk types.

The process you insert the chunk on is responsible for getting the data to the
correct output rank. It does so by placing the chunk in its **Outgoing** message
box and then working to send it (shown by the black lines connecting the processes).

Internally, the processes involved in a shuffle continuously

- receive newly inserted chunks from your program
- move chunks to their intended ranks
- receive chunks from other ranks
- hand off *ready* chunks when your program extracts them

During a shuffle, device memory might run low on more or more processes . `rapidsmpf` is able to *spill* chunks of data from device memory to a
larger pool (e.g. host memory). In the diagram above, this is shown by the
hatched chunks.

### Example: Shuffle a Table on a Column

The `rapidsmpf` Shuffler operates on **chunks** of data, without really caring
what those bytes represent. But one common use case is shuffling a table on (the
hash of) one or more columns. In this scenario, `rapidsmpf` can be used as part
of a Shuffle Join implementation.

This diagram shows multiple nodes working together to shuffle a large, logical
Table.

![A diagram showing how to use rapidsmpf to shuffle a table.](../_static/rapidsmpf-shuffle-table-fs8.png)

Suppose you have a large logical table that's split into a number of partitions.
In the diagram above, this is shown as the different dashed boxes on the
left-hand side. In this example, we've shown four partitions, but this could be
much larger. Each row in the table is assigned to some group (by the hash of the
columns you're joining on, say), which is shown by the color of the row.

Your program **inserts** data to the shuffler. In this case, it's inserting
chunks that represent pieces of the table that have been partitioned (by hash
key) and packed into a chunk.

Each rank involved in the shuffle knows which ranks are responsible for which
hash keys. For example, rank 1 knows that it's responsible for the purple
chunks, needs to send red chunks to rank 2, etc.

Each input partition possibly includes data for each hash key. All the processes
involved in the shuffle move data to get all the chunks with a particular hash
key to the correct rank (spilling if needed). This is shown in the middle
section.

As chunks become "ready" (see above), your program can **extract** chunks and
process them as necessary. This is shown on the right-hand side.

### Shuffle Statistics

Shuffles can be configured to collect statistics, which can help you understand the performance of the system.
This table gives an overview of the different statistics collected.

| Name | Type | Description |
| --- | --- | --- |
| `spill-bytes-device-to-host` | int | The size in bytes of data moved from device to host when spilling data. |
| `spill-time-device-to-host` | float | The duration of the device to host spill. The unit is platform dependent. |
| `spill-bytes-host-to-device` | int | The size in bytes of data moved from host to device when unspilling data. |
| `spill-time-host-to-device` | float | The duration of the host to device spill. The unit is platform dependent. |
| `spill-bytes-recv-to-host` | int | The size in bytes of data received into host memory on one node from some other node. |
| `shuffle-payload-send` | int | The size in bytes of data transferred from a node (including locally, from a node to itself). |
| `shuffle-payload-recv` | int | The size in bytes of data transferred to a node (including locally, from a node to itself). |
| `event-loop-total` | float | The duration of a Shuffler's event loop iteration. The unit is platform dependent. |
| `event-loop-metadata-send` | float | The duration of sending metadata from one node to another. The unit is platform dependent. |
| `event-loop-metadata-recv` | float | The duration of receiving any outstanding metadata messages from other nodes. The unit is platform dependent. |
| `event-loop-post-incoming-chunk-recv` | float | The duration of posting receives for any incoming chunks from other nodes. The unit is platform dependent. |
| `event-loop-init-gpu-data-send` | float | The duration of receiving ready-for-data messages and initiating data send operations. The duration of the actual data transfer is not captured by this statistic. The unit is platform dependent. |
| `event-loop-check-future-finish` | float | The duration spent checking if any data has finished being sent. The unit is platform dependent. |

Statistics are available in both C++ and [Python](#api-statistics).
