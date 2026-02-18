# Actors

{term}`Actor`s are coroutine-based asynchronous relational operators that read from
zero-or-more {term}`Channel`s and write to zero-or-more {term}`Channel`s within a {term}`Network`.

**C++**

```c++
// sum the row counts of all incoming table chunks
rapidsmpf::streaming::Actor accumulator(
    std::shared_ptr<streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::Channel> ch_in,
    std::shared_ptr<rapidsmpf::Channel> ch_out)
{
    co_await ctx->executor()->schedule();
    int64_t total = 0;
    while (true) {
        // continuously read until channel is empty
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            break;
        }
        auto chunk = co_await msg.release<rapidsmpf::streaming::TableChunk>()
                            .make_available(ctx);
        total += chunk.table_view().num_rows();

        // Forward the chunk downstream.
        co_await ch_out->send(
            streaming::to_message(msg.sequence_number(), std::move(chunk))
        );
    }
}
```

**Python**
```python
@define_actor()
async def accumulator(ctx, ch_in, ch_out):
    """Sum a column across all incoming table chunks."""

    total = 0
    msg: Message[TableChunk] | None
    # continuously read until channel is empty
    while (msg := await ch_in.recv(ctx)) is not None:
        # Convert the message into a TableChunk (releases the message).
        table = TableChunk.from_message(msg)
        total += table.table_view().num_rows()

        # Wrap and forward the chunk downstream.
        await ch_out.send(ctx, Message(msg.sequence_number, table))

    # Drain the output channel to close it gracefully.
    await ch_out.drain(ctx)
```

*examples of actors in C++ and Python*

## Actor Types

{term}`Actor`s fall into two categories:
- Local Actors: These include operations like filtering, projection, or column-wise transforms. They operate exclusively on local data and preserve CSP semantics.

- Collective Actors: These include {term}`Collective Operation`s like {term}`Shuffler`, join, groupby aggregations, etc. which require access to distributed data.

In the case of a collective {term}`Actor`, remote communication is handled internally. For example, a shuffle actor may need to access all {term}`Partition`s of a table, both local and remote, but this coordination and data exchange happens inside the CSP-process itself. As a reminder {term}`Channel`s are an abstraction and not used to serialize and pass data between workers.

This hybrid model, which combines a SPMD-style distribution model and a local CSP-style streaming model, offers several advantages:

- It enables clear process composition and streaming semantics for local operations.

- It allows collective coordination to be localized inside CSP-processes, avoiding the need for global synchronization or a complete global task graph.

- It makes inter-worker parallelism explicit through SPMD-style communication.

For examples of communication actors and collective operations please read the [shuffle architecture page](./shuffle-architecture.md).
