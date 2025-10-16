## Channels


Are asynchronous messaging queue used move messages between nodes.

┌─────────────────────────────────────────────────────────────────────────┐
│                          STREAMING NETWORK                              │
│                                                                         │
│  ┌──────────┐         ┌──────────┐         ┌──────────┐                 │
│  │  Node 1  │ ──ch1─> │  Node 2  │ ──ch2─> │  Node 3  │                 │
│  │(Producer)│         │(Transform)         │(Consumer)│                 │
│  └──────────┘         └──────────┘         └──────────┘                 │
│       │                    │                     │                      │
│    Message             Message               Message                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

*example streaming network with 3 Nodes and 2 Channels*

Components:
  • Node: Coroutine that processes messages
  • Channel: Async queue connecting nodes
  • Message: GPU Buffer with a CUDA Stream


Channels provide asynchronous communication with **backpressure**:

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

Consumer is **"full"** when an internal ring_buffer `coro::ring_buffer<Message, 1> rb_;` has reached capacity.  

Additional backpressure control can be applied by usage of a Throttline system (semaphores) controlling the number of threads/concurrent operations. 


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

Internally, when using a `throttle` a task that sends into a channel must acquire tickets granting permission to send before being able to send. The send then returns a receipt that grants permission to release the ticket.  The consumer of a throttled channel accepts messsages without issue.  This means that the throttle is localised to the producer tasks.

More simply, using a throttling adaptor limits the number tasks a producer sends into a channel.  This pattern is very useful for producer nodes where we want some amount of bounded concurrency in the tasks that might suspend before sending into a channel -- especially useful when trying to minimize the over-production of long-lived memory: reads/scans, shuffles, etc.



