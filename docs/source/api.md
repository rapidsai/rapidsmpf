# API Reference

This page contains the API reference for `rapidsmpf`.

## Integrations

The subpackages under `rapidsmpf.integrations` contain integrations with other
libraries.

### Generic

```{eval-rst}
.. automodule:: rapidsmpf.integrations
   :members:
```

(api-integration-dask)=
### Dask

```{eval-rst}
.. automodule:: rapidsmpf.integrations.dask
   :members:
```

### Single-process

```{eval-rst}
.. automodule:: rapidsmpf.integrations.single
   :members:
```

### Ray

```{eval-rst}
.. automodule:: rapidsmpf.integrations.ray
   :members:
```

### cuDF

```{eval-rst}
.. automodule:: rapidsmpf.integrations.cudf
   :members:
```

#### Partition

```{eval-rst}
.. autofunction:: rapidsmpf.integrations.cudf.partition.split_and_pack
.. autofunction:: rapidsmpf.integrations.cudf.partition.unpack_and_concat
```

## Shuffler

```{eval-rst}
.. automodule:: rapidsmpf.shuffler
   :members:
```

## Communicator

```{eval-rst}
.. automodule:: rapidsmpf.communicator
   :members:

.. automodule:: rapidsmpf.communicator.communicator
   :members:
```

### MPI Communicator

```{eval-rst}
.. automodule:: rapidsmpf.communicator.mpi
   :members:
```

### UCXX Communicator

```{eval-rst}
.. automodule:: rapidsmpf.communicator.ucxx
   :members:
```

## Buffer

```{eval-rst}
.. automodule:: rapidsmpf.memory
   :members:

.. automodule:: rapidsmpf.memory.buffer
   :members:

.. automodule:: rapidsmpf.memory.memory_reservation
   :members:

.. automodule:: rapidsmpf.memory.buffer_resource
   :members:

.. automodule:: rapidsmpf.memory.packed_data
   :members:

.. automodule:: rapidsmpf.memory.scoped_memory_record
   :members:
```

## Config Options

```{eval-rst}
.. automodule:: rapidsmpf.config
   :members:
```

(api-statistics)=
## Statistics

```{eval-rst}
.. automodule:: rapidsmpf.statistics
   :members:
```

## RMM Resource Adaptor

```{eval-rst}
.. automodule:: rapidsmpf.rmm_resource_adaptor
   :members:
```
