# Statistics

RapidsMPF can be configured to collect {term}`Statistics`, which can help you understand the performance of the system.
This table gives an overview of the different statistics collected.

| Name | Description |
| --- | --- |
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
| `spill-manager-limit-breach` | Shortfall statistics when the spill manager could not free enough memory. Reports the maximum, average, and count of breach events. Only recorded when spilling falls short of the requested amount. |

Statistics are available in both C++ and [Python](#api-statistics).

## Example Output

### Text (`report()`)

```
Statistics:
 - copy-device-to-pinned_host:           2.79 GiB | 473.25 ms | 5.90 GiB/s | avg-stream-delay 21.39 ms
 - copy-pinned_host-to-device:           2.79 GiB | 487.96 ms | 5.73 GiB/s | avg-stream-delay 26.74 ms
 - event-loop-check-future-finish:       715.55 us | avg 40.14 ns
 - event-loop-init-gpu-data-send:        715.54 us | avg 40.14 ns
 - event-loop-metadata-recv:             5.33 ms | avg 298.80 ns
 - event-loop-metadata-send:             1.88 ms | avg 105.46 ns
 - event-loop-post-incoming-chunk-recv:  613.30 us | avg 34.40 ns
 - event-loop-total:                     52.84 ms | avg 2.96 us
 - shuffle-payload-recv:                 2.79 GiB | avg 28.61 MiB
 - shuffle-payload-send:                 2.79 GiB | avg 28.61 MiB
 - spill-manager-limit-breach:           max 3.26 GiB | avg 2.93 GiB | count 1016
```

### JSON (`write_json()`)

JSON output contains raw numeric values for all statistics. Registered
formatters (which produce human-readable strings such as "1.0 KiB" or "3.5 ms"
in the text report) are not applied — values remain as plain numbers to keep
the output machine-parseable. For example, a bytes statistic that reads
`"3.00001e+09"` is three billion bytes; the text report would show `"2.79 GiB"`
for the same figure.

Raw units: memory sizes are in **bytes** (float), timings are in **seconds** (float).

```json
{
  "statistics": {
    "copy-device-to-pinned_host": {"count": 100, "value": 3.00001e+09, "max": 3.0029e+07},
    "copy-pinned_host-to-device": {"count": 100, "value": 3.00001e+09, "max": 3.0029e+07},
    "event-loop-check-future-finish": {"count": 17860, "value": 0.000602501, "max": 2.749e-06},
    "event-loop-init-gpu-data-send": {"count": 17860, "value": 0.000544818, "max": 1.854e-06},
    "event-loop-metadata-recv": {"count": 17860, "value": 0.00316116, "max": 0.000151427},
    "event-loop-metadata-send": {"count": 17860, "value": 0.00126099, "max": 2.318e-06},
    "event-loop-post-incoming-chunk-recv": {"count": 17860, "value": 0.00041276, "max": 2.286e-06},
    "event-loop-total": {"count": 17860, "value": 0.0416019, "max": 0.000181082},
    "shuffle-payload-recv": {"count": 100, "value": 3.00001e+09, "max": 3.0029e+07},
    "shuffle-payload-send": {"count": 100, "value": 3.00001e+09, "max": 3.0029e+07},
    "spill-manager-limit-breach": {"count": 988, "value": 3.11398e+12, "max": 3.49572e+09}
  },
  "memory_records": {
    "benchmarks/bench_shuffle.cpp:304(shuffling)": {"num_calls": 2, "peak_bytes": 600580452, "total_bytes": 28010554440, "global_peak_bytes": 600580452},
    "src/integrations/cudf/partition.cpp:126(split_and_pack)": {"num_calls": 20, "peak_bytes": 300029371, "total_bytes": 6000681620, "global_peak_bytes": 300029371},
    "src/integrations/cudf/partition.cpp:156(unpack_and_concat)": {"num_calls": 20, "peak_bytes": 300290532, "total_bytes": 6000043920, "global_peak_bytes": 300290532},
    "src/integrations/cudf/partition.cpp:91(partition_and_pack)": {"num_calls": 20, "peak_bytes": 600029371, "total_bytes": 16010491320, "global_peak_bytes": 600029371}
  }
}
```
