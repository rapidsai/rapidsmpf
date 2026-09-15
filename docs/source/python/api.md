# Python API Reference

This page contains the API reference for `rapidsmpf`.

## Buffers and memory utilities

```{eval-rst}
.. automodule:: rapidsmpf.memory
   :members:

.. automodule:: rapidsmpf.memory.buffer
   :members:

.. automodule:: rapidsmpf.memory.memory_reservation
   :members:

.. automodule:: rapidsmpf.memory.buffer_resource
   :members:

.. automodule:: rapidsmpf.memory.pinned_memory_resource
   :members:

.. automodule:: rapidsmpf.memory.packed_data
   :members:

.. automodule:: rapidsmpf.memory.scoped_memory_record
   :members:

.. automodule:: rapidsmpf.rmm_resource_adaptor
   :members:
```

## Multi-process Communicators

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

## Configuration options

```{eval-rst}
.. automodule:: rapidsmpf.config
   :members:
```

### Default configuration

```{eval-rst}
.. automodule:: rapidsmpf.config_defaults
   :members:
   :exclude-members: DEFAULTS

   .. py:data:: DEFAULTS
      :module: rapidsmpf.config_defaults
      :type: Mapping[str, str]

      Read-only mapping from known configuration keys to their default values.
```

(api-statistics)=
## Statistics

```{eval-rst}
.. automodule:: rapidsmpf.statistics
   :members:
```

## Collective primitives

```{eval-rst}
.. automodule:: rapidsmpf.coll
   :members:

.. automodule:: rapidsmpf.shuffler
   :members:
```

## Integrations

The subpackages under `rapidsmpf.integrations` contain integrations with other
libraries.

### Generic

```{eval-rst}
.. automodule:: rapidsmpf.integrations
   :members:
```

### Ray

```{eval-rst}
.. automodule:: rapidsmpf.integrations.ray
   :members:
```
