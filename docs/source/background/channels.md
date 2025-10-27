# Channels


Channels are asynchronous messaging queue used to move messages between {term}`Node`s in the rapidsmpf streaming network.

<img src="../_static/animation-legend.png" alt="Animation Legend" style="width: 320px;"/>
<img src="../_static/buffers-animated.gif" alt="Animated buffer pipeline" style="max-width: 4500px;"/>

<br/>
As buffers move through the graph, the channels (arrows) move from empty (dashed line) to full (solid line).


```
┌─────────────────────────────────────────────────────────────────────────┐
│                          STREAMING NETWORK                              │
│                                                                         │
│  ┌──────────┐         ┌──────────┐         ┌──────────┐                 │
│  │  Node 1  │ ──ch1─> │  Node 2  │ ──ch2─> │  Node 3  │                 │
│  │(Producer)│         │(Transform)         │(Consumer)│                 │
│  └──────────┘         └──────────┘         └──────────┘                 │
│       │                    │                     │                      │
│    Message              Message               Message                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```
*fig: example streaming network with 3 Nodes and 2 Channels*

Components:
  • Node: Coroutine that processes messages
  • Channel: Async queue connecting nodes
  • Message: GPU Buffer with a CUDA Stream

In the above graph, moving data in and out of channels on a single GPU should be relatively cheap, nearly free! This stratedy of using channels to move tasks/buffers is a core methodology for rapidsmpf to overlap: scans, compute, spilling, and communication.

## Backpressure

Channels provide asynchronous communication with **backpressure**.  Backpressure is built-in by limiting the number of buffers in a channel to a single slot

```
Producer Side:                    Consumer Side:
┌──────────────┐                 ┌──────────────┐
│   Producer   │                 │   Consumer   │
│              │                 │              │
│ send(msg) ───┼────> Channel ───┼───> receive()│
│   (async)    │     [buffer]    │    (async)   │
└──────────────┘                 └──────────────┘
      │                                  │
      │ If consumer is full,             │
      │ Suspends (backpressure)          │
      ▼                                  ▼
   Resumes when                      Operates when
   space available                   data available
```

Key Properties:
  • Non-blocking: Coroutines suspend, not threads
  • Backpressure: Slow consumers throttle producers
  • Type-safe: Messages are type-erased but validated

A Consumer is **"full"** when an internal ring_buffer `coro::ring_buffer<Message, 1> rb_;` has reached capacity.

Additional backpressure control can be applied by usage of a Throttling
system (semaphores) controlling the maximum number of
threads/concurrent operations.


```python
throttle = asyncio.Semaphore(4)
async with throttle:
      msg = Message(chunk)
      await ch_out.send(msg)
```


```c++
auto throttle = std::make_shared<ThrottlingAdaptor>(ch, 4);
std::vector<Node> producers;]
constexpr int n_producer{100};
for (int i = 0; i < n_producer; i++) {
    producers.push_back(producer(ctx, throttle, i));
}
```

Internally, when using a `throttle` a Node that writes into a channel must acquire a ticket granting permission to write before being able to. The write/send then returns a receipt that grants permission to release the ticket.  The consumer of a throttled channel reads messages without issue.  This means that the throttle is localised to the producer nodes.

More simply, using a throttling adaptor limits the number messages a producer writes into a channel.  This pattern is very useful for producer nodes where we want some amount of bounded concurrency in the tasks that might suspend before sending into a channel -- especially useful when trying to minimize the over-production of long-lived memory: reads/scans, shuffles, etc.

eg. a source node that read files. `ThrottlingAdaptor` will allow the node to delay reading files, until it has acquired a ticket to send a message to the channel. In comparison, non-throttling channels will suspend during send by which time, the files have already loaded into the memory unnecessarily
