# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference as deref
from cython.operator cimport preincrement
from libcpp cimport bool as bool_t
from libcpp.memory cimport make_shared, make_unique, shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector

from dataclasses import dataclass

from rapidsmpf.config cimport Options, cpp_Options
from rapidsmpf.memory.scoped_memory_record cimport ScopedMemoryRecord
from rapidsmpf.rmm_resource_adaptor cimport (RmmResourceAdaptor,
                                             cpp_RmmResourceAdaptor)


cdef extern from "<rapidsmpf/statistics.hpp>" nogil:
    cdef shared_ptr[cpp_Statistics] cpp_from_options \
        "rapidsmpf::Statistics::from_options"(
            cpp_RmmResourceAdaptor* mr, cpp_Options options
        ) except +


cdef extern from *:
    """
    std::size_t cpp_get_statistic_count(
        rapidsmpf::Statistics const& stats, std::string const& name
    ) {
        return stats.get_stat(name).count();
    }
    double cpp_get_statistic_value(
        rapidsmpf::Statistics const& stats, std::string const& name
    ) {
        return stats.get_stat(name).value();
    }
    std::vector<std::string> cpp_list_stat_names(rapidsmpf::Statistics const& stats) {
        return stats.list_stat_names();
    }
    void cpp_clear_statistics(rapidsmpf::Statistics& stats) {
        stats.clear();
    }
    """
    size_t cpp_get_statistic_count(cpp_Statistics stats, string name) except + nogil
    double cpp_get_statistic_value(cpp_Statistics stats, string name) except + nogil
    vector[string] cpp_list_stat_names(cpp_Statistics stats) except + nogil
    void cpp_clear_statistics(cpp_Statistics stats) except + nogil

cdef class Statistics:
    """
    Track statistics across RapidsMPF operations.

    Parameters
    ----------
    enable
        Whether statistics tracking is enabled.
    mr
        Enable memory profiling by providing a RMM resource adaptor.
    """
    def __init__(self, *, bool_t enable, RmmResourceAdaptor mr = None):
        cdef cpp_RmmResourceAdaptor* mr_handle
        self._mr = mr
        if enable and mr is not None:
            mr_handle = mr.get_handle()
            with nogil:
                self._handle = make_shared[cpp_Statistics](mr_handle)
        else:
            with nogil:
                self._handle = make_shared[cpp_Statistics](enable)

    @classmethod
    def from_options(cls, RmmResourceAdaptor mr not None, Options options not None):
        """
        Construct from configuration options.

        Parameters
        ----------
        mr
            Pointer to a memory resource used for memory profiling.
        options
            Configuration options.

        Returns
        -------
        The constructed Statistics instance.
        """
        cdef Statistics ret = cls.__new__(cls)
        cdef cpp_RmmResourceAdaptor* mr_handle = mr.get_handle()
        with nogil:
            ret._handle = cpp_from_options(mr_handle, options._handle)
        ret._mr = mr
        return ret

    def __dealloc__(self):
        with nogil:
            self._handle.reset()

    @property
    def enabled(self):
        """
        Checks if statistics is enabled.

        Operations on disabled statistics is no-ops.

        Returns
        -------
        True if statistics is enabled, otherwise False.
        """
        return deref(self._handle).enabled()

    def report(self):
        """
        Generates a report of statistics in a formatted string.

        Operations on disabled statistics is no-ops.

        Returns
        -------
        A string representing the formatted statistics report.
        """
        cdef string ret
        with nogil:
            ret = deref(self._handle).report()
        return ret.decode('UTF-8')

    def get_stat(self, name):
        """
        Retrieves a statistic by name.

        Parameters
        ----------
        name
            Name of the statistic to retrieve.

        Returns
        -------
        A dict of the statistic.

        Raises
        ------
        KeyError
            If the statistic with the specified name does not exist.
        """
        cdef string name_ = str.encode(name)
        cdef size_t count
        cdef double value
        try:
            with nogil:
                count = cpp_get_statistic_count(deref(self._handle), name_)
                value = cpp_get_statistic_value(deref(self._handle), name_)
        except IndexError:
            # The C++ implementation throws a std::out_of_range exception
            # which we / Cython translate to a KeyError.
            raise KeyError(f"Statistic '{name}' does not exist") from None
        return {"count": count, "value": value}

    def list_stat_names(self):
        """
        Returns a list of all statistic names.
        """
        cdef vector[string] names = cpp_list_stat_names(deref(self._handle))
        cdef vector[string].iterator it = names.begin()
        ret = []
        while it != names.end():
            ret.append(deref(it).decode("utf-8"))
            preincrement(it)
        return ret

    def add_stat(self, name, double value):
        """
        Adds a value to a statistic.

        Parameters
        ----------
        name
            Name of the statistic.
        value
            Value to add.

        Returns
        -------
        Updated total value.
        """
        cdef string name_ = str.encode(name)
        cdef double ret
        with nogil:
            ret = deref(self._handle).add_stat(name_, value)
        return ret

    @property
    def memory_profiling_enabled(self):
        """
        Checks if memory profiling is enabled.

        Returns
        -------
        True if memory profiling is enabled, otherwise False.
        """
        return deref(self._handle).is_memory_profiling_enabled()

    def get_memory_records(self):
        """
        Retrieves all memory profiling records stored by this instance.

        Returns
        -------
        Dictionary mapping record names to memory usage data.
        """
        cdef unordered_map[string, cpp_MemoryRecord] records
        with nogil:
            records = deref(self._handle).get_memory_records()

        cdef unordered_map[string, cpp_MemoryRecord].iterator it = records.begin()
        ret = {}
        while it != records.end():
            name = deref(it).first.decode("utf-8")
            ret[name] = create_memory_record_from_cpp(deref(it).second)
            preincrement(it)
        return ret

    def memory_profiling(self, name):
        """
        Create a scoped memory profiling context for a named code region.

        Returns a context manager that tracks memory allocations and
        deallocations made through the associated memory resource while
        the context is active. The profiling data is aggregated under
        the provided ``name`` and made available via
        :meth:`Statistics.get_memory_records()`.

        The statistics include:
        - Total and peak memory allocated within the scope (``scoped``)
        - Global peak memory usage during the scope (``global_peak``)
        - Number of times the named scope was entered (``num_calls``)

        If memory profiling is disabled or the memory resource is ``None``,
        this is a no-op.

        Parameters
        ----------
        name
            A unique identifier for the profiling scope. Used as a key
            when accessing profiling data via :meth:`Statistics.get_memory_records`.

        Returns
        -------
        A context manager that collects memory profiling data.

        Examples
        --------
        >>> import rmm
        >>> mr = RmmResourceAdaptor(rmm.mr.CudaMemoryResource())
        >>> stats = Statistics(enable=True, mr=mr)
        >>> with stats.memory_profiling("outer"):
        ...     b1 = rmm.DeviceBuffer(size=1024, mr=mr)
        ...     with stats.memory_profiling("inner"):
        ...         b2 = rmm.DeviceBuffer(size=1024, mr=mr)
        >>> inner = stats.get_memory_records()["inner"]
        >>> print(inner.scoped.peak())
        1024
        >>> outer = stats.get_memory_records()["outer"]
        >>> print(outer.scoped.peak())
        2048
        """
        return MemoryRecorder(self, self._mr, name)

    def clear(self) -> None:
        """
        Clears all statistics.

        Memory profiling records are not cleared.
        """
        with nogil:
            cpp_clear_statistics(deref(self._handle))


@dataclass
class MemoryRecord:
    """
    Holds memory profiling statistics for a named scope.

    Attributes
    ----------
    scoped
        Memory statistics collected while the scope was active, including
        number of allocations, peak bytes allocated, and total allocated bytes.
    global_peak
        The maximum global memory usage observed during the scope,
        including allocations from other threads or nested scopes.
    num_calls
        Number of times the profiling context with this name was entered.
    """
    scoped: ScopedMemoryRecord
    global_peak: int
    num_calls: int


cdef create_memory_record_from_cpp(cpp_MemoryRecord handle):
    """Help function to create a MemoryRecord from a cpp_MemoryRecord"""
    return MemoryRecord(
        scoped=ScopedMemoryRecord.from_handle(handle.scoped),
        global_peak=handle.global_peak,
        num_calls=handle.num_calls,
    )


cdef class MemoryRecorder:
    """
    A context manager for recording memory allocation statistics within a code block.

    This class is not intended to be used directly by end users. Instead, use
    :meth:`Statistics.memory_profiling`, which creates and manages an instance
    of this class.

    Parameters
    ----------
    stats
        The statistics object responsible for aggregating memory profiling data.
    mr
        The memory resource through which allocations are tracked.
    name
        The name of the profiling scope. Used as a key in the statistics record.
    """
    def __cinit__(
        self, Statistics stats not None, RmmResourceAdaptor mr not None, name
    ):
        self._stats = stats
        self._mr = mr
        self._name = str.encode(name)

    def __enter__(self):
        if self._mr is None:
            return

        cdef cpp_RmmResourceAdaptor* mr = self._mr.get_handle()
        with nogil:
            self._handle = make_unique[cpp_MemoryRecorder](
                self._stats._handle.get(), mr, self._name
            )

    def __exit__(self, exc_type, exc_value, traceback):
        if self._mr is not None:
            with nogil:
                self._handle.reset()
        return False  # do not suppress exceptions
