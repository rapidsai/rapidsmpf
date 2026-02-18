# Quickstart


## Shuffle Basics

`rapidsmpf` is designed as a service that plugs into other libraries. This means
it isn't typically used as a standalone library, and is expected to operate in
some larger runtime.

### Dask-cuDF Example

`rapidsmpf` can be used with [Dask-cuDF] to shuffle a Dask DataFrame. This toy
example just loads the shuffled data into GPU memory. In practice, you would
reduce the output or write it to disk after shuffling.

```python
import dask.distributed
import dask.dataframe as dd
from dask_cuda import LocalCUDACluster

from rapidsmpf.examples.dask import dask_cudf_shuffle


df = dask.datasets.timeseries().reset_index(drop=True).to_backend("cudf")

# RapidsMPF is compatible with `dask_cuda` workers.
# Use an rmm pool for optimal performance.
with LocalCUDACluster(rmm_pool_size=0.8) as cluster:
    with dask.distributed.Client(cluster) as client:
        shuffled = dask_cudf_shuffle(df, on=["name"])

        # collect the results in memory.
        result = shuffled.compute()
```

After shuffling on `name`, all of the records with a particular name will be in
the same partition. See [Dask Integration](#api-integration-dask) for more.

[Dask-cuDF]: https://docs.rapids.ai/api/dask-cudf/stable/

## Streaming Engine

Basic streaming pipeline example in Python.  In this example we have 3 {term}`Actor`s
in the {term}`Network`: push_to_channel->count_num_rows->pull_from_channel.

*note: push_to_channel/pull_from_channel are convenience functions which simulate scans/writes*

```{literalinclude} ../../python/rapidsmpf/rapidsmpf/examples/streaming/basic_example.py
:language: python
:lines: 34-
```
