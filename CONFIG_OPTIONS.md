# RapidsMPF Configuration Options

RapidsMPF can be configured using a dictionary of options, which may be populated via environment variables. All dictionary keys are automatically converted to **lowercase**.

Each configuration option includes:

- **Name**: The key used in the configuration dictionary.
- **Environment Variable**: The corresponding environment variable name.
- **Description**: Describes what the option controls, including accepted values.

> [!NOTE]
> Environment variable names are always uppercase and prefixed with `RAPIDSMPF_`.
>
> Typically, it is up to the user to read environment variables using code such as:
>
> ```python
> options = Options()
> options.insert_if_absent(get_environment_variables())
> ```
>
> However, Dask automatically reads environment variables for any options not set explicitly when calling `bootstrap_dask_cluster()`.
>
> It is always explicit in C++, use something like:
> ```c++
>   rapidsmpf::config::Options options{rapidsmpf::config::get_environment_variables()};
> ```

---

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


### Dask Integration

- **`dask_spill_device`**
  - **Environment Variable**: `RAPIDSMPF_DASK_SPILL_DEVICE`
  - **Default**: `0.50`
  - **Description**: GPU memory limit for shuffling as a fraction of total device memory.

- **`dask_oom_protection`**
  - **Environment Variable**: `RAPIDSMPF_DASK_OOM_PROTECTION`
  - **Default**: `False`
  - **Description**: Enable out-of-memory protection by using managed memory when the device
  memory pool raises OOM errors.

- **`dask_periodic_spill_check`**
  - **Environment Variable**: `RAPIDSMPF_DASK_PERIODIC_SPILL_CHECK`
  - **Default**: `1e-3`
  - **Description**: Enable periodic spill checks. A dedicated thread continuously
  checks and perform spilling based on the current available memory as reported by
  the buffer resource. The value of `dask_periodic_spill_check` is used as the pause
  between checks (in seconds). Use `"disabled"` to disable periodic spill checks.

- **`dask_statistics`**
  - **Environment Variable**: `RAPIDSMPF_DASK_STATISTICS`
  - **Default**: `False`
  - **Description**: Enable RapidsMPF statitistics, which will be printed by each Worker
  on shutdown.
