# API Reference

This page contains the API reference for `rapidsmpf`.

## Integrations

The subpackages under `rapidsmpf.integrations` contain integrations with other
libraries.

### Core

```{eval-rst}
.. automodule:: rapidsmpf.integrations.core
   :members:
```

### Single

```{eval-rst}
.. automodule:: rapidsmpf.integrations.single
   :members:
```

### Dask

```{eval-rst}
.. automodule:: rapidsmpf.integrations.dask
   :members:
```

(integration-examples-dask)=
#### Dask Examples

```{eval-rst}
.. automodule:: rapidsmpf.examples.dask
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

### Single Communicator

```{eval-rst}
.. automodule:: rapidsmpf.communicator.single
   :members:
```

## Buffer

```{eval-rst}
.. automodule:: rapidsmpf.buffer
   :members:

.. automodule:: rapidsmpf.buffer.buffer
   :members:

.. automodule:: rapidsmpf.buffer.resource
   :members:

.. automodule:: rapidsmpf.buffer.packed_data
   :members:
```

## Config Options

```{eval-rst}
.. automodule:: rapidsmpf.config
   :members:
