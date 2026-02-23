# Statistics

RapidsMPF can be configured to collect {term}`Statistics`, which can help you understand the performance of the system.
This table gives an overview of the different statistics collected.

| Name | Description |
| --- | --- |
| `copy-{src}-to-{dst}` | Amount of data copied between memory types by RapidsMPF. `{src}` and `{dst}` are `device`, `pinned_host`, or `host`. |
| `event-loop-check-future-finish` | Time spent polling for completed data transfers. |
| `event-loop-init-gpu-data-send` | Time spent initiating GPU data sends. Does not include actual transfer time. |
| `event-loop-metadata-recv` | Time spent receiving chunk metadata from other ranks. |
| `event-loop-metadata-send` | Time spent sending chunk metadata to other ranks. |
| `event-loop-post-incoming-chunk-recv` | Time spent posting receive buffers for incoming chunk data. |
| `event-loop-total` | Time spent in one Shuffler event-loop iteration. |
| `recv-into-host-memory` | Data received directly into host memory rather than device memory, due to memory pressure at receive time. |
| `shuffle-payload-recv` | Shuffle data received by this rank, including self-transfers. |
| `shuffle-payload-send` | Shuffle data sent from this rank, including self-transfers. |
| `spill-manager-limit-breach` | Average shortfall when the spill manager could not free enough memory. Only recorded when spilling falls short of the requested amount. |

Statistics are available in both C++ and [Python](#api-statistics).

## Example Output

```
Statistics:
 - copy-device-to-device:                2.79 GiB (avg 28.61 MiB)
 - copy-pinned_host-to-pinned_host:      2.79 GiB (avg 28.61 MiB)
 - event-loop-check-future-finish:       462.50 us (avg 24.97 ns)
 - event-loop-init-gpu-data-send:        546.70 us (avg 29.52 ns)
 - event-loop-metadata-recv:             2.81 ms (avg 151.47 ns)
 - event-loop-metadata-send:             1.20 ms (avg 64.81 ns)
 - event-loop-post-incoming-chunk-recv:  415.83 us (avg 22.45 ns)
 - event-loop-total:                     32.55 ms (avg 1.76 us)
 - shuffle-payload-recv:                 2.79 GiB (avg 28.61 MiB)
 - shuffle-payload-send:                 2.79 GiB (avg 28.61 MiB)
 - spill-manager-limit-breach:           avg 2.93 GiB
```
