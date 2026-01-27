# RapidsMPF

Collection of multi-GPU, distributed memory algorithms.

## Getting started

Building rapidsmpf from source is recommended when running a nightly/upstream versions, since dependencies on non-ABI-stable libraries (e.g., pylibcudf) could cause temporary breakage leading to issues such as segmentation faults. Stable versions can be installed from conda or pip packages.

### Build from source

Clone rapidsmpf and install the dependencies in a conda environment:
```bash
git clone https://github.com/rapidsai/rapidsmpf.git
cd rapidsmpf

# Choose a environment file that match your system.
mamba env create --name rapidsmpf-dev --file conda/environments/all_cuda-131_arch-$(uname -m).yaml

# Build
./build.sh
```

#### Debug build

Debug builds can be produced by adding the `-g` flag:

```bash
./build.sh -g
```

##### AddressSanitizer-enabled build

Enabling the [AddressSanitizer](https://github.com/google/sanitizers/wiki/AddressSanitizer) is also possible with the `--asan` flag:

```bash
./build.sh -g --asan
```

C++ code built with AddressSanitizer should simply work, but there are caveats for CUDA and Python code. Any CUDA code executing with AddressSanitizer requires `protect_shadow_gap=0`, which can be set via an environment variable:

```bash
ASAN_OPTIONS=protect_shadow_gap=0
```

On the other hand, Python may require `LD_PRELOAD` to be set so that the AddressSanitizer is loaded before Python. On a conda environment, for example, there is usually a `$CONDA_PREFIX/lib/libasan.so`, and thus the application may be launched as follows:

```bash
LD_PRELOAD=$CONDA_PREFIX/lib/libasan.so python ...
```

Python applications using CUDA will require setting both environment variables described above

### MPI

Run the test suite using MPI:
```bash
# When using OpenMP, we need to enable CUDA support.
export OMPI_MCA_opal_cuda_support=1

# Run the suite using two MPI processes.
mpirun -np 2 cpp/build/gtests/mpi_tests

# Alternatively
cd cpp/build && ctest -R mpi_tests_2
```

We can also run the shuffle benchmark. To assign each MPI rank its own GPU, we use a [binder script](https://github.com/LStuber/binding/blob/master/binder.sh):
```bash
# The binder script requires numactl `mamba install numactl`
wget https://raw.githubusercontent.com/LStuber/binding/refs/heads/master/binder.sh
chmod a+x binder.sh
mpirun -np 2 ./binder.sh cpp/build/benchmarks/bench_shuffle
```

### UCX

The UCX test suite uses, for convenience, MPI to bootstrap, therefore we need to launch UCX tests with `mpirun`. Run the test suite using UCX:
```bash
# Run the suite using two processes.
mpirun -np 2 cpp/build/gtests/ucxx_tests
```

### rrun - Distributed Launcher

RapidsMPF includes `rrun`, a lightweight launcher that eliminates the MPI dependency for multi-GPU workloads. This is particularly useful for development, testing, and environments where MPI is not available.

#### Single-Node Usage

```bash
# Build rrun
cd cpp/build
cmake --build . --target rrun

# Launch 2 ranks in the local node
./tools/rrun -n 2 ./benchmarks/bench_comm -C ucxx -O all-to-all

# With verbose output and specific GPUs
./tools/rrun -v -n 4 -g 0,1,2,3 ./benchmarks/bench_comm -C ucxx
```

## Algorithms
### Table Shuffle Service
Example of a MPI program that uses the shuffler:
```cpp
#include <vector>

#include <mpi.h>
#include <unistd.h>

#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/communicator/mpi.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/integrations/cudf/partition.hpp>
#include <rapidsmpf/shuffler/shuffler.hpp>

#include "../benchmarks/utils/random_data.hpp"

// An example of how to use the shuffler.
int main(int argc, char** argv) {
    // In this example we use the MPI backed. For convenience, rapidsmpf provides an
    // optional MPI-init function that initialize MPI with thread support.
    rapidsmpf::mpi::init(&argc, &argv);

    // Initialize configuration options from environment variables.
    rapidsmpf::config::Options options{rapidsmpf::config::get_environment_variables()};

    // First, we have to create a Communicator, which we will use throughout the
    // example. Notice, if you want to do multiple shuffles concurrently, each shuffle
    // should use its own Communicator backed by its own MPI communicator.
    std::shared_ptr<rapidsmpf::Communicator> comm =
        std::make_shared<rapidsmpf::MPI>(MPI_COMM_WORLD, options);

    // Create a statistics instance for the shuffler that tracks useful information.
    auto stats = std::make_shared<rapidsmpf::Statistics>();

    // Then a progress thread where the shuffler event loop executes is created. A single
    // progress thread may be used by multiple shufflers simultaneously.
    std::shared_ptr<rapidsmpf::ProgressThread> progress_thread =
        std::make_shared<rapidsmpf::ProgressThread>(comm->logger(), stats);

    // The Communicator provides a logger.
    auto& log = comm->logger();

    // We will use the same stream, memory, and buffer resource throughout the example.
    rmm::cuda_stream_view stream = cudf::get_default_stream();
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref();
    rapidsmpf::BufferResource br{mr};

    // As input data, we use a helper function from the benchmark suite. It creates a
    // random cudf table with 2 columns and 100 rows. In this example, each MPI rank
    // creates its own local input and we only have one input per rank but each rank
    // could take any number of inputs.
    cudf::table local_input = random_table(2, 100, 0, 10, stream, mr);

    // The total number of inputs equals the number of ranks, in this case.
    auto const total_num_partitions =
        static_cast<rapidsmpf::shuffler::PartID>(comm->nranks());

    // We create a new shuffler instance, which represents a single shuffle. It takes
    // a Communicator, the total number of partitions, and a "owner function", which
    // map partitions to their destination ranks. All ranks must use the same owner
    // function, in this example we use the included round-robin owner function.
    rapidsmpf::shuffler::Shuffler shuffler(
        comm,
        progress_thread,
        0,  // op_id
        total_num_partitions,
        &br,
        stats,
        rapidsmpf::shuffler::Shuffler::round_robin  // partition owner
    );

    // It is our own responsibility to partition and pack (serialize) the input for
    // the shuffle. The shuffler only handles raw host and device buffers. However, it
    // does provide a convenience function that hash partitions a cudf table and packs
    // each partition. The result is a mapping of `PartID`, globally unique partition
    // identifiers, to their packed partitions.
    std::unordered_map<rapidsmpf::shuffler::PartID, rapidsmpf::PackedData> packed_inputs =
        rapidsmpf::partition_and_pack(
            local_input,
            {0},  // columns_to_hash
            static_cast<int>(total_num_partitions),
            cudf::hash_id::HASH_MURMUR3,
            cudf::DEFAULT_HASH_SEED,
            stream,
            &br
        );

    // Now, we can insert the packed partitions into the shuffler. This operation is
    // non-blocking and we can continue inserting new input partitions. E.g., a pipeline
    // could read, hash-partition, pack, and insert, one parquet-file at a time while the
    // distributed shuffle is being processed underneath.
    shuffler.insert(std::move(packed_inputs));

    // When we are finished inserting to a specific partition, we tell the shuffler.
    // Again, this is non-blocking and should be done as soon as we known that we don't
    // have more inputs for a specific partition. In this case, we are finished with all
    // partitions.
    for (rapidsmpf::shuffler::PartID i = 0; i < total_num_partitions; ++i) {
        shuffler.insert_finished(i);
    }

    // Vector to hold the local results of the shuffle operation.
    std::vector<std::unique_ptr<cudf::table>> local_outputs;

    // Wait for and process the shuffle results for each partition.
    while (!shuffler.finished()) {
        // Block until a partition is ready and retrieve its partition ID.
        rapidsmpf::shuffler::PartID finished_partition = shuffler.wait_any();

        // Extract the finished partition's data from the Shuffler.
        auto packed_chunks = shuffler.extract(finished_partition);

        // Unpack (deserialize) and concatenate the chunks into a single table using a
        // convenience function.
        local_outputs.push_back(
            rapidsmpf::unpack_and_concat(
                rapidsmpf::unspill_partitions(std::move(packed_chunks), &br, true),
                stream,
                &br
            )
        );
    }
    // At this point, `local_outputs` contains the local result of the shuffle.
    // Let's log the result.
    log.print("Finished shuffle with ", local_outputs.size(), " local output partitions");

    // Log the statistics report.
    log.print(stats->report());

    // Shutdown the Shuffler explicitly or let it go out of scope for cleanup.
    shuffler.shutdown();

    // Finalize the execution, `RAPIDSMPF_MPI` is a convenience macro that
    // checks for MPI errors.
    RAPIDSMPF_MPI(MPI_Finalize());
}
```

## RapidsMPF Configuration Options

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

### Available Options

#### General

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


#### Dask Integration

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
  - **Description**: Enable RapidsMPF statitistics collection.

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
