# Statistics

RapidsMPF can be configured to collect {term}`Statistics`, which can help you understand the performance of the system.
This table gives an overview of the different statistics collected.

| Name | Description |
| --- | --- |
| `alloc-{memtype}` | Bytes allocated via `BufferResource::allocate()`, broken down by memory type (`device`, `pinned_host`, `host`). Shows total bytes, total time, allocation throughput, and average stream delay. |
| `copy-{src}-to-{dst}` | Amount of data copied between memory types by RapidsMPF. `{src}` and `{dst}` are `device`, `pinned_host`, or `host`. Shows total bytes, total copy time, throughput, and average stream delay (time between CPU submission and GPU execution of the copy). |
| `event-loop-check-future-finish` | Time spent polling for completed data transfers. |
| `event-loop-init-gpu-data-send` | Time spent initiating GPU data sends. Does not include actual transfer time. |
| `event-loop-metadata-recv` | Time spent receiving chunk metadata from other ranks. |
| `event-loop-metadata-send` | Time spent sending chunk metadata to other ranks. |
| `event-loop-post-incoming-chunk-recv` | Time spent posting receive buffers for incoming chunk data. |
| `event-loop-total` | Time spent in one Shuffler event-loop iteration. |
| `recv-into-host-memory` | Data received directly into host memory rather than device memory, due to memory pressure at receive time. |
| `shuffle-payload-recv` | Shuffle data received by this rank, including self-transfers. |
| `shuffle-payload-send` | Shuffle data sent from this rank, including self-transfers. |

Statistics are available in both C++ and [Python](#api-statistics).

## Example Output

### Text (`report()`)

```
Statistics:
 - alloc-device:                         2.79 GiB | 198.84 us | 13.72 TiB/s | avg-stream-delay 26.44 ms
 - alloc-pinned_host:                    2.79 GiB | 244.62 us | 11.15 TiB/s | avg-stream-delay 21.07 ms
 - copy-device-to-pinned_host:           2.79 GiB | 467.16 ms | 5.98 GiB/s | avg-stream-delay 21.06 ms
 - copy-pinned_host-to-device:           2.79 GiB | 481.25 ms | 5.81 GiB/s | avg-stream-delay 26.44 ms
 - event-loop-check-future-finish:       548.01 us | avg 30.79 ns
 - event-loop-init-gpu-data-send:        609.03 us | avg 34.21 ns
 - event-loop-metadata-recv:             3.54 ms | avg 199.06 ns
 - event-loop-metadata-send:             1.41 ms | avg 79.16 ns
 - event-loop-post-incoming-chunk-recv:  514.04 us | avg 28.88 ns
 - event-loop-total:                     49.16 ms | avg 2.76 us
 - shuffle-payload-recv:                 2.79 GiB | avg 28.61 MiB
 - shuffle-payload-send:                 2.79 GiB | avg 28.61 MiB
```

### JSON (`write_json()`)

JSON output contains raw numeric values for all statistics. Registered
formatters (which produce human-readable strings such as "1.0 KiB" or "3.5 ms"
in the text report) are not applied — values remain as plain numbers to keep
the output machine-parseable. For example, a bytes statistic that reads
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
    "event-loop-check-future-finish": {"count": 17800, "value": 0.00054801, "max": 2.8e-06},
    "event-loop-init-gpu-data-send": {"count": 17800, "value": 0.00060903, "max": 2.0e-06},
    "event-loop-metadata-recv": {"count": 17800, "value": 0.00354, "max": 1.5e-04},
    "event-loop-metadata-send": {"count": 17800, "value": 0.00141, "max": 2.3e-06},
    "event-loop-post-incoming-chunk-recv": {"count": 17800, "value": 0.00051404, "max": 2.3e-06},
    "event-loop-total": {"count": 17800, "value": 0.04916, "max": 1.8e-04},
    "shuffle-payload-recv": {"count": 100, "value": 2.9957e+09, "max": 3.0029e+07},
    "shuffle-payload-send": {"count": 100, "value": 2.9957e+09, "max": 3.0029e+07}
  },
}
```
