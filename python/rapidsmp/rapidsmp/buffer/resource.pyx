# Copyright (c) 2025, NVIDIA CORPORATION.

from cython.operator cimport dereference as deref
from libc.stdint cimport int64_t
from libcpp.cast cimport dynamic_cast
from libcpp.memory cimport make_shared
from rmm.librmm.memory_resource cimport (device_memory_resource,
                                         statistics_resource_adaptor)
from rmm.pylibrmm.memory_resource cimport (DeviceMemoryResource,
                                           StatisticsResourceAdaptor)


cdef class BufferResource:
    """
    A buffer resource.

    Parameters
    ----------
    device_mr
        Reference to the RMM device memory resource used for device allocations.
    """
    def __cinit__(self, DeviceMemoryResource device_mr, memory_available):
        self._handle = make_shared[cpp_BufferResource](device_mr.get_mr())
        cdef unordered_map[MemoryType, cpp_MemoryAvailable] _mem_available

    cdef cpp_BufferResource* ptr(self):
        """
        A raw pointer to the underlying C++ `BufferResource`.

        Returns
        -------
            The raw pointer.
        """
        return self._handle.get()

# Alias of a `rmm::mr::statistics_resource_adaptor` pointer.
ctypedef statistics_resource_adaptor[device_memory_resource]* stats_mr_ptr

cdef class LimitAvailableMemory:
    """
    A callback class for querying the remaining available memory within a defined limit
    from an RMM statistics resource adaptor.

    This class is primarily designed to simulate constrained memory environments
    or prevent memory allocation beyond a specific threshold. It provides
    information about the available memory by subtracting the memory currently
    used (as reported by the RMM statistics resource adaptor) from a user-defined limit.

    It is typically used in the context of memory management operations such as
    with `BufferResource`.

    Parameters
    ----------
    statistics_mr
        A statistics resource adaptor that tracks memory usage and provides
        statistics about the memory consumption. The `LimitAvailableMemory`
        instance keeps a reference to `statistics_mr` to keep it alive.
    limit
        The maximum memory limit (in bytes). Used to calculate the remaining
        available memory.

    Notes
    -----
    - The `statistics_mr` resource must not be destroyed while this object is
      still in use.

    Examples
    --------
    >>> from your_module import LimitAvailableMemory, StatisticsResourceAdaptor
    >>> stats_mr = StatisticsResourceAdaptor(...)
    >>> memory_limiter = LimitAvailableMemory(stats_mr, limit=1_000_000)
    """
    def __init__(self, StatisticsResourceAdaptor statistics_mr, int64_t limit):
        self._statistics_mr = statistics_mr  # Keep the mr alive.
        cdef stats_mr_ptr mr = dynamic_cast[stats_mr_ptr](statistics_mr.get_mr())
        assert mr  # The dynamic cast should always succeed.
        self._handle = make_shared[cpp_LimitAvailableMemory](mr, limit)

    def __call__(self):
        """
        Returns the remaining available memory within the defined limit.

        This method queries the `rmm_statistics_resource` to determine the memory
        currently in use and calculates the remaining memory as:
        `limit - used_memory`.

        Returns
        -------
        The remaining memory in bytes.
        """
        cdef int64_t ret
        with nogil:
            ret = deref(self._handle)()
        return ret
