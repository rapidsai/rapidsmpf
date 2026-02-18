# Actors

{term}`Actor`s are coroutine-based asynchronous relational operators that read from
zero-or-more {term}`Channel`s and write to zero-or-more {term}`Channel`s within a {term}`Network`.

**C++**

```c++
// sum a column
rapidsmpf::streaming::Actor accumulator(
    std::shared_ptr<rapidsmpf::Channel> ch_out,
    std::shared_ptr<rapidsmpf::Channel> ch_in)
{
    int64_t total = 0;
    while (true) {
        // continuously read until channel is empty
        auto msg = co_await ch_in->recv();
        if (!msg) {
            break;
        }

        auto column = ... // get column from data buffer in message

        total += column->sum<int64_t>();
    }

    // Send the accumulated result downstream as a message
    co_await ch_out->send(total));
}
```

**Python**

```python
async def accumulator(ch_out, ch_in, msg):
    """Sum Column"""

    total = 0
    # continuously read until channel is empty
    while (msg := await ch_in is not None:)
        col = ... # get column from data buffer in message
        total += sum(col)

    # Send the accumulated result downstream as a message
    send(total, ch_out)
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
