## Nodes

Nodes are coroutine-based asynchronous relational operators that perform operations on data.  Messages (data buffer) is read in and written to via [channels](./channels.md)


**Python**

```python
async def accumulator(ctx: Context, ch_out: Channel, ch_in: Channel):
    """Sum Column"""

    total = 0
    while (msg := await ch_in is not None:)
        chunk = TableChunk.from_message(msg)
        total = SUM(chunk.column)
    await send(total)
```

**C++**

```c++
# sum a column
rapidsmpf::task<void> accumulator(
    std::shared_ptr<rapidsmpf::Channel> ch_out,
    std::shared_ptr<rapidsmpf::Channel> ch_in)
{
    int64_t total = 0;
    while (true) {
        auto msg = co_await ch_in->recv();
        if (!msg) {
            break;
        }
        auto chunk = rapidsmpf::TableChunk::from_message(*msg);

        total += chunk->get_column("value")->sum<int64_t>();
    }

    // Send the accumulated result downstream as a message
    co_await ch_out->send(rapidsmpf::make_message(total));
}
```

##$ Node Types

Nodes fall into two categories:
- Local Nodes: These include operations like filtering, projection, or column-wise transforms. They operate exclusively on local data and preserve CSP semantics.

- Collective Nodes: These include operations like shuffle, join, groupby aggregrations, etc. which require access to distributed data. 

In the case of a collective nodes, remote communication is handled internally. For example, a shuffle node may need to access all partitions of a table, both local and remote, but this coordination and data exchange happens inside the CSP-process itself.  As a reminder "Channels" are an abstraction and not used to serialize and pass data between workers

This hybrid model, which combines a SPMD-style distribution model and a local CSP-style streaming model, offers several advantages:

- It enables clear process composition and streaming semantics for local operations.

- It allows collective coordination to be localized inside CSP-processes, avoiding the need for global synchronization or a complete global task graph.

- It makes inter-worker parallelism explicit through SPMD-style communication.

For examples of communication nodes please read the [shuffle architecture page](./shuffle-architecture.md)