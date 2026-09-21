# Statistics

RapidsMPF can be configured to collect {term}`Statistics`, which can help you understand the performance of the system.
This table gives an overview of the different statistics collected.

| Name | Description |
| --- | --- |
| `alloc-{memtype}` | Bytes allocated via `BufferResource::allocate()`, broken down by memory type (`device`, `pinned_host`, `host`). Shows total bytes, total time, allocation throughput, and average stream delay. |
| `copy-{src}-to-{dst}` | Amount of data copied between memory types by RapidsMPF. `{src}` and `{dst}` are `device`, `pinned_host`, or `host`. Shows total bytes, total copy time, throughput, and average stream delay (time between CPU submission and GPU execution of the copy). |
| `buffer-spilled-time` | How long spilled data remains spilled until unspilled back into device memory. |
| `event-loop-total` | Time spent in in the background `ProgressThread` event-loop. |
| `recv-into-host-memory` | Data received directly into host memory rather than device memory, due to memory pressure at receive time. |
| `reserve-{memtype}-wait-avoided` | Reservation requests that `MemoryReserveOrWait` satisfied at once, over all requests. A miss means the request had to wait. |
| `reserve-{memtype}-wait-timeout` | Waiting requests that ran out `memory_reserve_timeout`, over the requests that had to wait. A miss means memory was released in time. |
| `reserve-{memtype}-wait-satisfied-time` | Time requests spent waiting before a reservation release satisfied them. |
| `reserve-{memtype}-wait-timeout-time` | Time requests spent waiting before the progress timeout fired. |
| `reserve-{memtype}-waiting-requests` | Requests waiting concurrently. Recorded each time a request starts waiting, not sampled over time, so the maximum is exact while the mean is the queue depth seen when a request starts waiting. |
| `reserve-{memtype}-request-bytes` | Bytes requested from `reserve_or_wait()`. |
| `reserve-{memtype}-overbook-bytes` | Bytes by which `reserve_or_wait_or_overbook()` exceeded the memory limit after the timeout, counting only what each request added rather than the total outstanding deficit. |
| `shuffle-payload-recv` | Shuffle data received by this rank, excluding self-transfers. |
| `shuffle-payload-send` | Shuffle data sent from this rank, excluding self-transfers. |
| `allgather-payload-recv` | AllGather data received by this rank, excluding self-transfers. |
| `allgather-payload-send` | AllGather data sent from this rank, excluding self-transfers. |
| `sparsealltoall-payload-recv` | SparseAlltoall data received by this rank, excluding self-transfers. |
| `sparsealltoall-payload-send` | SparseAlltoall data sent from this rank, excluding self-transfers. |
| `allreduce-payload-recv` | AllReduce data received by this rank, excluding self-transfers. |
| `allreduce-payload-send` | AllReduce data sent from this rank, excluding self-transfers. |

Statistics are available in both C++ and [Python](#api-statistics).

## Example Output

### Text (`report()`)

```
Statistics:
 - alloc-device:                         2.79 GiB | 198.84 us | 13.72 TiB/s | avg-stream-delay 26.44 ms
 - alloc-pinned_host:                    2.79 GiB | 244.62 us | 11.15 TiB/s | avg-stream-delay 21.07 ms
 - copy-device-to-pinned_host:           2.79 GiB | 467.16 ms | 5.98 GiB/s | avg-stream-delay 21.06 ms
 - copy-pinned_host-to-device:           2.79 GiB | 481.25 ms | 5.81 GiB/s | avg-stream-delay 26.44 ms
 - event-loop-total:                     49.16 ms | avg 2.76 us
 - reserve-device-overbook-bytes:        512.00 MiB | avg 42.67 MiB
 - reserve-device-request-bytes:         2.79 GiB | avg 28.61 MiB
 - reserve-device-wait-avoided:          73/100 (hits/lookups)
 - reserve-device-wait-satisfied-time:   41.20 ms | avg 2.75 ms
 - reserve-device-wait-timeout:          12/27 (hits/lookups)
 - reserve-device-wait-timeout-time:     1.20 s | avg 100.01 ms
 - reserve-device-waiting-requests:      max 4 | avg 1.8 (27 samples)
 - shuffle-payload-recv:                 2.79 GiB | avg 28.61 MiB
 - shuffle-payload-send:                 2.79 GiB | avg 28.61 MiB
```

### JSON (`write_json()`)

JSON output contains raw numeric values for all statistics. Formatters
(which produce human-readable strings such as "1.0 KiB" or "3.5 ms" in the
text report) are not applied — values remain as plain numbers to keep the
output machine-parseable. For example, a bytes statistic that reads
`"2.9957e+09"` is roughly three billion bytes; the text report would show `"2.79 GiB"`
for the same figure.

Raw units: memory sizes are in **bytes** (float), timings are in **seconds** (float).

```json
{
  "statistics": {
    "alloc-device-bytes": {"count": 100, "value": 2.9957e+09, "max": 3.0029e+07},
    "alloc-device-stream-delay": {"count": 100, "value": 2.644, "max": 2.7e-02},
    "alloc-device-time": {"count": 100, "value": 0.00019884, "max": 2.0e-06},
    "alloc-pinned_host-bytes": {"count": 100, "value": 2.9957e+09, "max": 3.0029e+07},
    "alloc-pinned_host-stream-delay": {"count": 100, "value": 2.107, "max": 2.2e-02},
    "alloc-pinned_host-time": {"count": 100, "value": 0.00024462, "max": 2.5e-06},
    "copy-device-to-pinned_host-bytes": {"count": 100, "value": 2.9957e+09, "max": 3.0029e+07},
    "copy-device-to-pinned_host-stream-delay": {"count": 100, "value": 2.106, "max": 2.2e-02},
    "copy-device-to-pinned_host-time": {"count": 100, "value": 0.46716, "max": 5.0e-03},
    "copy-pinned_host-to-device-bytes": {"count": 100, "value": 2.9957e+09, "max": 3.0029e+07},
    "copy-pinned_host-to-device-stream-delay": {"count": 100, "value": 2.644, "max": 2.7e-02},
    "copy-pinned_host-to-device-time": {"count": 100, "value": 0.48125, "max": 5.1e-03},
    "event-loop-total": {"count": 17800, "value": 0.04916, "max": 1.8e-04},
    "reserve-device-overbook-bytes": {"count": 12, "value": 5.3687e+08, "max": 6.7109e+07},
    "reserve-device-request-bytes": {"count": 100, "value": 2.9957e+09, "max": 3.0029e+07},
    "reserve-device-wait-avoided": {"count": 100, "value": 73, "max": 1},
    "reserve-device-wait-satisfied-time": {"count": 15, "value": 0.04120, "max": 8.1e-03},
    "reserve-device-wait-timeout": {"count": 27, "value": 12, "max": 1},
    "reserve-device-wait-timeout-time": {"count": 12, "value": 1.20012, "max": 1.0012e-01},
    "reserve-device-waiting-requests": {"count": 27, "value": 48.6, "max": 4},
    "shuffle-payload-recv": {"count": 100, "value": 2.9957e+09, "max": 3.0029e+07},
    "shuffle-payload-send": {"count": 100, "value": 2.9957e+09, "max": 3.0029e+07}
  },
}
```
