# Channels


{term}`Channel`s are asynchronous messaging queues used to move {term}`Message`s between {term}`Actor`s in the rapidsmpf streaming {term}`Network`.

```{image} ../_static/animation-legend.png
:width: 320px
:alt: Animation Legend
```

```{image} ../_static/buffers-animated.gif
:width: 4500px
:alt: Animated buffer pipeline
```

<br/>
As buffers move through the network, the channels (arrows) move from empty (dashed line) to full (solid line).


```
┌─────────────────────────────────────────────────────────────────────────┐
│                          STREAMING NETWORK                              │
│                                                                         │
│  ┌──────────┐         ┌──────────┐         ┌──────────┐                 │
│  │  Actor 1 │ ──ch1─> │  Actor 2 │ ──ch2─> │  Actor 3 │                 │
│  │(Producer)│         │(Transform)         │(Consumer)│                 │
│  └──────────┘         └──────────┘         └──────────┘                 │
│       │                    │                     │                      │
│    Message              Message               Message                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```
*fig: example streaming network with 3 Actors and 2 Channels*

Components:
  • {term}`Actor`: Coroutine that processes messages
  • {term}`Channel`: Async queue connecting actors
  • {term}`Message`: GPU {term}`Buffer` with a CUDA Stream

In the above network, moving data in and out of {term}`Channel`s on a single GPU should be relatively cheap, nearly free! This strategy of using channels to move tasks/{term}`Buffer`s is a core methodology for rapidsmpf to overlap: scans, compute, {term}`Spilling`, and communication.

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
  • Type-safe: {term}`Message`s are type-erased but validated

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
std::vector<Actor> producers;]
constexpr int n_producer{100};
for (int i = 0; i < n_producer; i++) {
    producers.push_back(producer(ctx, throttle, i));
}
```

Internally, when using a `throttle` an {term}`Actor` that writes into a {term}`Channel` must acquire a ticket granting permission to write before being able to. The write/send then returns a receipt that grants permission to release the ticket. The consumer of a throttled channel reads {term}`Message`s without issue. This means that the throttle is localised to the producer actors.

More simply, using a throttling adaptor limits the number of {term}`Message`s a producer writes into a {term}`Channel`. This pattern is very useful for producer {term}`Actor`s where we want some amount of bounded concurrency in the tasks that might suspend before sending into a channel -- especially useful when trying to minimize the over-production of long-lived memory: reads/scans, shuffles, etc.

e.g. a source actor that reads files. `ThrottlingAdaptor` will allow the {term}`Actor` to delay reading files, until it has acquired a ticket to send a {term}`Message` to the {term}`Channel`. In comparison, non-throttling channels will suspend during send by which time, the files have already loaded into the memory unnecessarily.
