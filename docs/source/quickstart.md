# Quickstart

This page introduces the basics of a shuffle using `rapidsmp`.

`rapidsmp` is designed as a service that plugs into other libraries. This means
it isn't typically used as a standalone library, and is expected to operate in
some larger runtime.

## Dask-cuDF Example

`rapidsmp` can be used with [Dask-cuDF] to shuffle a Dask DataFrame. This toy
example just loads the shuffled data into GPU memory. In practice, you would
reduce the output or write it to disk after shuffling.

```python
import dask.distributed
import dask.dataframe as dd
from dask_cuda import LocalCUDACluster

from rapidsmp.examples.dask import dask_cudf_shuffle


df = dask.datasets.timeseries().reset_index(drop=True).to_backend("cudf")

with LocalCUDACluster() as cluster:
    with dask.distributed.Client(cluster) as client:
        shuffled = dask_cudf_shuffle(df, shuffle_on=["name"])

        # collect the results in memory.
        result = shuffled.compute()
```

After shuffling on `name`, all of the records with a particular name will be in
the same partition. See [Dask Integration](#api-integration-dask) for more.

[Dask-cuDF]: https://docs.rapids.ai/api/dask-cudf/stable/
