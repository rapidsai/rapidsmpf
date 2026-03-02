# Configuration Options

RapidsMPF can be configured using a dictionary of options, which may be populated via
environment variables. All dictionary keys are automatically converted to **lowercase**.
See the [Python API Reference](python/api.md) for full details.

Each configuration option includes:

- **Name**: The key used in the configuration dictionary.
- **Environment Variable**: The corresponding environment variable name.
- **Description**: Describes what the option controls, including accepted values.

Environment variable names are always uppercase and prefixed with `RAPIDSMPF_`.

Typically, it is up to the user to read environment variables using code such as:

```python
options = Options()
options.insert_if_absent(get_environment_variables())
```

However, Dask automatically reads environment variables for any options not set
explicitly when calling {func}`rapidsmpf.integrations.dask.bootstrap_dask_cluster`.

It is always explicit in C++, use something like:

```c++
rapidsmpf::config::Options options{rapidsmpf::config::get_environment_variables()};
```

## Available Options

### General

- **`log`**
  - **Environment Variable**: `RAPIDSMPF_LOG`
  - **Default**: `WARN`
  - **Description**: Controls the logging verbosity level. Valid values are:
    - `NONE`: Disable all logging.
    - `PRINT`: General print messages.
    - `WARN`: Warning messages (**default**).
    - `INFO`: Informational messages.
    - `DEBUG`: Debug-level messages.
    - `TRACE`: Fine-grained trace-level messages.

- **`statistics`**
  - **Environment Variable**: `RAPIDSMPF_STATISTICS`
  - **Default**: `False`
  - **Description**: Enable RapidsMPF statistics collection.

- **`num_streaming_threads`**
  - **Environment Variable**: `RAPIDSMPF_NUM_STREAMING_THREADS`
  - **Default**: `1`
  - **Description**: Number of threads used to execute coroutines. Must be greater than zero.

- **`num_streams`**
  - **Environment Variable**: `RAPIDSMPF_NUM_STREAMS`
  - **Default**: `16`
  - **Description**: Number of CUDA streams used by RapidsMPF. A pool of CUDA
    streams is created at startup, and work is scheduled onto these streams to
    enable concurrent GPU execution and overlap of computation and data movement.
    Must be greater than zero.

- **`memory_reserve_timeout`**
  - **Environment Variable**: `RAPIDSMPF_MEMORY_RESERVE_TIMEOUT`
  - **Default**: `100 ms`
  - **Description**: Controls the global progress timeout for memory reservation
    requests. If the value does not include a unit, it is interpreted as seconds.

    The value limits how long the system may go without making progress on any
    pending memory reservation. When the timeout expires and no reservation has
    been satisfied, the system forces progress by selecting a pending request and
    attempting to reserve memory for it. Depending on the context, this may
    result in an empty reservation, an overbooked reservation, or a failure.

    This option ensures forward progress under memory pressure and prevents the
    system from stalling indefinitely when memory availability fluctuates.

- **`allow_overbooking_by_default`**
  - **Environment Variable**: `RAPIDSMPF_ALLOW_OVERBOOKING_BY_DEFAULT`
  - **Default**: `true`
  - **Description**: Controls the default overbooking behavior for *high-level*
    memory reservation APIs, such as `reserve_memory()`.

    When enabled, high-level memory reservation requests may overbook memory
    after the global `memory_reserve_timeout` expires, allowing forward
    progress under memory pressure.

    When disabled, high-level memory reservation requests fail with an error if
    no progress is possible within the timeout.

    This option is only used when a high-level API does not explicitly specify
    an overbooking policy. It does **not** change the behavior of lower-level
    memory reservation primitives or imply that overbooking is enabled or
    disabled globally across the system.

- **`pinned_memory`**
  - **Environment Variable**: `RAPIDSMPF_PINNED_MEMORY`
  - **Default**: `false`
  - **Description**: Enables pinned host memory if it is available on the system.
    Pinned host memory provides higher bandwidth and lower latency for device-to-host
    transfers compared to regular pageable host memory. When enabled, RapidsMPF
    primarily uses pinned host memory for spilling. Availability of pinned host memory
    can be checked using `is_pinned_memory_resources_supported()`.

- **`spill_device_limit`**
  - **Environment Variable**: `RAPIDSMPF_SPILL_DEVICE_LIMIT`
  - **Default**: `80%`
  - **Description**: Soft upper limit on device memory usage that RapidsMPF attempts
    to stay under by triggering spilling. This limit is a best-effort target and may
    not always be enforceable. The value can be specified either as an absolute byte
    count (e.g. `"10GiB"`, `"512MB"`) or as a percentage of the total memory of the
    current device (e.g. `"80%"`).

- **`periodic_spill_check`**
  - **Environment Variable**: `RAPIDSMPF_PERIODIC_SPILL_CHECK`
  - **Default**: `1ms`
  - **Description**: Enable periodic spill checks. A dedicated thread continuously
    checks and performs spilling based on the current available memory as reported by
    the buffer resource. The value of `periodic_spill_check` specifies the pause
    between checks and supports time units, e.g. `us` or `ms`. If no unit is
    specified, seconds are assumed. Use `"disabled"` to disable periodic spill checks.

- **`unbounded_file_read_cache`**
  - **Environment Variable**: `RAPIDSMPF_UNBOUNDED_FILE_READ_CACHE`
  - **Default**: `"disabled"`
  - **Description**: Configure caching of file read results for file-backed messages.

    When set to a memory type (for example `host`, `pinned`, or `device`), the
    first read of a specific file slice is cached by storing a copy of the data
    in the associated Context's message storage using the specified memory type.
    Subsequent reads of the exact same slice reuse the cached copy instead of
    re-reading from disk.

    When set to `"disabled"`, no caching is performed.

    This option is primarily intended for benchmarking and performance analysis.
    After an initial warm-up run, subsequent runs can avoid most disk I/O (metadata
    access and file listing will still occur).

    Cached data is scoped to the Context that performed the read, and its lifetime
    matches the lifetime of that Context. The cache is unbounded, and entries are
    only released when the Context is destroyed.

    **Warning:** This feature assumes that each file slice is always read with
    identical read parameters, such as filters and schemas. No validation is
    performed. If these parameters differ, incorrect data may be returned.

### Dask Integration

- **`dask_spill_device`**
  - **Environment Variable**: `RAPIDSMPF_DASK_SPILL_DEVICE`
  - **Default**: `0.50`
  - **Description**: GPU memory limit for shuffling as a fraction of total device memory.

- **`dask_spill_to_pinned_memory`**
  - **Environment Variable**: `RAPIDSMPF_DASK_SPILL_TO_PINNED_MEMORY`
  - **Default**: `False`
  - **Description**: Control whether RapidsMPF spills to pinned host memory when
    available, or falls back to regular pageable host memory. Pinned host memory
    provides higher bandwidth and lower latency for device-to-host transfers
    compared to pageable host memory.

- **`dask_oom_protection`**
  - **Environment Variable**: `RAPIDSMPF_DASK_OOM_PROTECTION`
  - **Default**: `False`
  - **Description**: Enable out-of-memory protection by using managed memory when
    the device memory pool raises OOM errors.

- **`dask_periodic_spill_check`**
  - **Environment Variable**: `RAPIDSMPF_DASK_PERIODIC_SPILL_CHECK`
  - **Default**: `1e-3`
  - **Description**: Enable periodic spill checks. A dedicated thread continuously
    checks and performs spilling based on the current available memory as reported
    by the buffer resource. The value of `dask_periodic_spill_check` is used as the
    pause between checks (in seconds). Use `"disabled"` to disable periodic spill
    checks.

- **`dask_statistics`**
  - **Environment Variable**: `RAPIDSMPF_DASK_STATISTICS`
  - **Default**: `False`
  - **Description**: Enable RapidsMPF statistics collection.

- **`dask_print_statistics`**
  - **Environment Variable**: `RAPIDSMPF_DASK_STATISTICS`
  - **Default**: `True`
  - **Description**: Print RapidsMPF statistics to stdout on Dask Worker shutdown
    when `dask_statistics` is enabled.

- **`dask_staging_spill_buffer`**
  - **Environment Variable**: `RAPIDSMPF_DASK_STAGING_SPILL_BUFFER`
  - **Default**: `128 MiB`
  - **Description**: Size of the intermediate staging buffer (in bytes) used for
    device-to-host spilling. This temporary buffer is allocated on the device to
    reduce memory pressure when transferring Python-managed GPU objects during
    Dask spilling. Use `disabled` to skip allocation of the staging buffer.
