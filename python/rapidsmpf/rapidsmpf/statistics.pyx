# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference as deref
from cython.operator cimport preincrement
from libcpp cimport bool
from libcpp.memory cimport make_shared, make_unique
from libcpp.string cimport string

from dataclasses import dataclass

from rapidsmpf.rmm_resource_adaptor cimport (RmmResourceAdaptor,
                                             ScopedMemoryRecord,
                                             cpp_RmmResourceAdaptor)


# Since `Statistics::Stat` doesn't have a default ctor, we use the following
# getters.
cdef extern from *:
    """
    std::size_t cpp_get_statistic_count(
        rapidsmpf::Statistics const& stats, std::string const& name
    ) {
        return stats.get_stat(name).count();
    }
    std::size_t cpp_get_statistic_value(
        rapidsmpf::Statistics const& stats, std::string const& name
    ) {
        return stats.get_stat(name).value();
    }
    """
    size_t cpp_get_statistic_count(cpp_Statistics stats, string name) nogil
    double cpp_get_statistic_value(cpp_Statistics stats, string name) nogil

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
    def __cinit__(self, *, bool enable, RmmResourceAdaptor mr = None):
        cdef cpp_RmmResourceAdaptor* mr_handle
        self._mr = mr  # Keep mr alive.
        if enable and mr is not None:
            mr_handle = mr.get_handle()
            with nogil:
                self._handle = make_shared[cpp_Statistics](mr_handle)
        else:
            with nogil:
                self._handle = make_shared[cpp_Statistics](enable)

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
        True if statistics is enabled, otherwise false.
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
        with nogil:
            count = cpp_get_statistic_count(deref(self._handle), name_)
            value = cpp_get_statistic_value(deref(self._handle), name_)
        return {"count": count, "value": value}

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
        True if memory profiling is enabled, otherwise false.
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
        return MemoryRecorder(self, self._mr, name)


@dataclass
class MemoryRecord:
    scoped: ScopedMemoryRecord
    global_peak: int
    num_calls: int


cdef create_memory_record_from_cpp(cpp_MemoryRecord handle):
    """Help function to create a MemoryRecord from a cpp_MemoryRecord"""
    return MemoryRecord(
        scoped = ScopedMemoryRecord.from_handle(handle.scoped),
        global_peak = handle.global_peak,
        num_calls = handle.num_calls
    )


cdef class MemoryRecorder:
    def __cinit__(self, Statistics stats, RmmResourceAdaptor mr, name):
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
