# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference as deref
from libc.stdint cimport int64_t
from libcpp.cast cimport dynamic_cast
from libcpp.memory cimport make_shared, shared_ptr
from libcpp.utility cimport move
from rmm.librmm.memory_resource cimport (device_memory_resource,
                                         statistics_resource_adaptor)
from rmm.pylibrmm.memory_resource cimport (DeviceMemoryResource,
                                           StatisticsResourceAdaptor)


# Converter from `shared_ptr[cpp_LimitAvailableMemory]` to `cpp_MemoryAvailable`
cdef extern from *:
    """
    std::function<std::int64_t()> to_MemoryAvailable(
        std::shared_ptr<rapidsmpf::LimitAvailableMemory> functor
    ) {
        return *functor;
    }
    """
    cpp_MemoryAvailable to_MemoryAvailable(
        shared_ptr[cpp_LimitAvailableMemory]
    ) except +


cdef class BufferResource:
    """
    Class managing buffer resources.

    This class handles memory allocation and transfers between different memory types
    (e.g., host and device). All memory operations in RapidsMPF, such as those performed
    by the Shuffler, rely on a buffer resource for memory management.

    Parameters
    ----------
    device_mr
        Reference to the RMM device memory resource used for device allocations.
    memory_available
        Optional memory availability functions. Memory types without availability
        functions are unlimited. A function must return the current available
        memory of a specific type. It must be thread-safe if used by multiple
        `BufferResource` instances concurrently.
        Warning: calling any `BufferResource` instance methods within the function
        might result in a deadlock. This is because the buffer resource is locked
        when the function is called.
    periodic_spill_check
        Enable periodic spill checks. A dedicated thread continuously checks and
        perform spilling based on the memory availability functions. The value of
        ``periodic_spill_check`` is used as the pause between checks (in seconds).
        If None, no periodic spill check is performed.
    """
    def __cinit__(
        self,
        DeviceMemoryResource device_mr,
        memory_available = None,
        periodic_spill_check = 1e-3
    ):
        cdef unordered_map[MemoryType, cpp_MemoryAvailable] _mem_available
        if memory_available is not None:
            for mem_type, func in memory_available.items():
                if not isinstance(func, LimitAvailableMemory):
                    raise NotImplementedError(
                        "Currently, BufferResource only accept `LimitAvailableMemory` "
                        "as memory available functions."
                    )
                _mem_available[<MemoryType?>mem_type] = to_MemoryAvailable(
                    (<LimitAvailableMemory?>func)._handle
                )
        cdef optional[cpp_Duration] period
        if periodic_spill_check is not None:
            period = cpp_Duration(periodic_spill_check)

        with nogil:
            self._handle = make_shared[cpp_BufferResource](
                device_mr.get_mr(),
                move(_mem_available),
                period,
            )
        self.spill_manager = SpillManager._create(self)

    def __dealloc__(self):
        """
        Deallocate resource without holding the GIL.

        This is important to ensure owned resources, like the underlying C++
        `SpillManager` object is destroyed, ensuring any threads can be
        joined without risk of deadlocks if both thread compete for the GIL.
        """
        with nogil:
            self._handle.reset()

    cdef cpp_BufferResource* ptr(self):
        """
        A raw pointer to the underlying C++ `BufferResource`.

        Returns
        -------
            The raw pointer.
        """
        return self._handle.get()

    def memory_reserved(self, MemoryType mem_type):
        """
        Get the current reserved memory of the specified memory type.

        Parameters
        ----------
        mem_type
            The target memory type.

        Returns
        -------
        The memory reserved, in bytes.
        """
        cdef size_t ret
        with nogil:
            ret = deref(self._handle).cpp_memory_reserved(mem_type)
        return ret


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
        instance keeps a reference to ``statistics_mr`` to keep it alive.
    limit
        The maximum memory limit (in bytes). Used to calculate the remaining
        available memory.

    Notes
    -----
    The ``statistics_mr`` resource must not be destroyed while this object is
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
        with nogil:
            self._handle = make_shared[cpp_LimitAvailableMemory](mr, limit)

    def __call__(self):
        """
        Returns the remaining available memory within the defined limit.

        This method queries the ``rmm_statistics_resource`` to determine the memory
        currently in use and calculates the remaining memory as:
        ``limit - used_memory``.

        Returns
        -------
        int
            The remaining memory in bytes.
        """
        cdef int64_t ret
        with nogil:
            ret = deref(self._handle)()
        return ret

    def __dealloc__(self):
        with nogil:
            self._handle.reset()
