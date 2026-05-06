# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cpython.bytes cimport PyBytes_FromStringAndSize
from cython.operator cimport dereference as deref
from cython.operator cimport preincrement
from libc.stdint cimport uint8_t
from libc.string cimport memcpy
from libcpp cimport bool as bool_t
from libcpp.memory cimport make_shared, make_unique, shared_ptr
from libcpp.optional cimport optional
from libcpp.string cimport string
from libcpp.vector cimport vector

import json
from dataclasses import dataclass

from rapidsmpf._detail.exception_handling cimport ex_handler
from rapidsmpf.config cimport Options, cpp_Options
from rapidsmpf.memory.pinned_memory_resource cimport (PinnedMemoryResource,
                                                      cpp_PinnedMemoryResource)
from rapidsmpf.memory.scoped_memory_record cimport ScopedMemoryRecord
from rapidsmpf.rmm_resource_adaptor cimport (RmmResourceAdaptor,
                                             cpp_RmmResourceAdaptor)

import os


cdef extern from "<rapidsmpf/statistics.hpp>" nogil:
    cdef shared_ptr[cpp_Statistics] cpp_from_options \
        "rapidsmpf::Statistics::from_options"(
            cpp_Options options,
        ) except +ex_handler


cdef extern from *:
    """
    #include <filesystem>
    #include <optional>
    #include <sstream>
    std::string cpp_report(
        rapidsmpf::Statistics const& stats,
        rapidsmpf::RmmResourceAdaptor* mr_ptr,
        std::optional<rapidsmpf::PinnedMemoryResource> const& pinned_mr
    ) {
        std::optional<rapidsmpf::RmmResourceAdaptor> mr =
            mr_ptr ? std::make_optional(*mr_ptr) : std::nullopt;
        return stats.report({.mr = std::move(mr), .pinned_mr = pinned_mr});
    }
    std::string cpp_report(
        rapidsmpf::Statistics const& stats,
        rapidsmpf::RmmResourceAdaptor* mr_ptr,
        std::optional<rapidsmpf::PinnedMemoryResource> const& pinned_mr,
        std::string const& header
    ) {
        std::optional<rapidsmpf::RmmResourceAdaptor> mr =
            mr_ptr ? std::make_optional(*mr_ptr) : std::nullopt;
        return stats.report({.mr = std::move(mr), .pinned_mr = pinned_mr,
            .header = header});
    }
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
    double cpp_get_statistic_max(
        rapidsmpf::Statistics const& stats, std::string const& name
    ) {
        return stats.get_stat(name).max();
    }
    std::vector<std::string> cpp_list_stat_names(rapidsmpf::Statistics const& stats) {
        return stats.list_stat_names();
    }
    void cpp_clear_statistics(rapidsmpf::Statistics& stats) {
        stats.clear();
    }
    void cpp_write_json(
        rapidsmpf::Statistics const& stats, std::string const& filepath
    ) {
        stats.write_json(std::filesystem::path(filepath));
    }
    std::string cpp_write_json_string(rapidsmpf::Statistics const& stats) {
        std::ostringstream ss;
        stats.write_json(ss);
        return ss.str();
    }
    // Wrap the span-based Statistics::deserialize so Cython can pass a vector.
    std::shared_ptr<rapidsmpf::Statistics> cpp_deserialize_statistics(
        std::vector<std::uint8_t> const& v
    ) {
        return rapidsmpf::Statistics::deserialize(
            std::span<std::uint8_t const>(v.data(), v.size())
        );
    }
    // Wrap the span-based Statistics::merge so Cython can pass a vector.
    std::shared_ptr<rapidsmpf::Statistics> cpp_merge_statistics(
        std::vector<std::shared_ptr<rapidsmpf::Statistics>> const& v
    ) {
        return rapidsmpf::Statistics::merge(
            std::span<std::shared_ptr<rapidsmpf::Statistics> const>(
                v.data(), v.size()
            )
        );
    }
    """
    string cpp_report(
        cpp_Statistics stats,
        cpp_RmmResourceAdaptor* mr_ptr,
        optional[cpp_PinnedMemoryResource] pinned_mr,
    ) except +ex_handler nogil
    string cpp_report(
        cpp_Statistics stats,
        cpp_RmmResourceAdaptor* mr_ptr,
        optional[cpp_PinnedMemoryResource] pinned_mr,
        string header,
    ) except +ex_handler nogil
    size_t cpp_get_statistic_count(cpp_Statistics stats, string name) \
        except +ex_handler nogil
    double cpp_get_statistic_value(cpp_Statistics stats, string name) \
        except +ex_handler nogil
    double cpp_get_statistic_max(cpp_Statistics stats, string name) \
        except +ex_handler nogil
    vector[string] cpp_list_stat_names(cpp_Statistics stats) except +ex_handler nogil
    void cpp_clear_statistics(cpp_Statistics stats) except +ex_handler nogil
    void cpp_write_json(cpp_Statistics stats, string filepath) \
        except +ex_handler nogil
    string cpp_write_json_string(cpp_Statistics stats) except +ex_handler nogil
    shared_ptr[cpp_Statistics] cpp_deserialize_statistics(
        const vector[uint8_t]& v
    ) except +ex_handler nogil
    shared_ptr[cpp_Statistics] cpp_merge_statistics(
        const vector[shared_ptr[cpp_Statistics]]& v
    ) except +ex_handler nogil

cdef class Statistics:
    """
    Track statistics across RapidsMPF operations.

    Parameters
    ----------
    enable
        Whether statistics tracking is enabled.
    """
    def __init__(self, *, bool_t enable):
        with nogil:
            self._handle = make_shared[cpp_Statistics](enable)

    @classmethod
    def from_options(cls, Options options not None):
        """
        Construct from configuration options.

        Parameters
        ----------
        options
            Configuration options.

        Returns
        -------
        The constructed Statistics instance.
        """
        cdef Statistics ret = cls.__new__(cls)
        with nogil:
            ret._handle = cpp_from_options(options._handle)
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

    def report(
        self,
        *,
        RmmResourceAdaptor mr = None,
        PinnedMemoryResource pinned_mr = None,
        header = None,
    ):
        """
        Generates a report of statistics in a formatted string.

        Operations on disabled statistics is no-ops.

        Parameters
        ----------
        mr
            When provided, a memory profiling section is included in the
            report. When ``None``, the memory profiling section shows
            "Disabled".
        pinned_mr
            When provided, a pinned memory section is included in the
            report.
        header
            Header line prepended to the report. When ``None``, the C++
            default is used.

        Returns
        -------
        A string representing the formatted statistics report.
        """
        cdef string ret
        cdef cpp_RmmResourceAdaptor* mr_ptr = NULL
        cdef optional[cpp_PinnedMemoryResource] cpp_pinned
        cdef string cpp_header
        if mr is not None:
            mr_ptr = mr.get_handle()
        if pinned_mr is not None:
            cpp_pinned = pinned_mr._handle
        if header is None:
            with nogil:
                ret = cpp_report(deref(self._handle), mr_ptr, cpp_pinned)
        else:
            cpp_header = header.encode()
            with nogil:
                ret = cpp_report(deref(self._handle), mr_ptr, cpp_pinned, cpp_header)
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
        cdef double max_val
        try:
            with nogil:
                count = cpp_get_statistic_count(deref(self._handle), name_)
                value = cpp_get_statistic_value(deref(self._handle), name_)
                max_val = cpp_get_statistic_max(deref(self._handle), name_)
        except IndexError:
            # The C++ implementation throws a std::out_of_range exception
            # which we / Cython translate to a KeyError.
            raise KeyError(f"Statistic '{name}' does not exist") from None
        return {"count": count, "value": value, "max": max_val}

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

    def to_dict(self):
        """
        Return a plain dict snapshot of all statistics.

        Each entry maps a stat name to a dict with ``count``, ``value``,
        and ``max`` keys, matching the shape returned by :meth:`get_stat`.
        The snapshot is taken atomically and is detached thus mutating it
        does not affect the underlying :class:`Statistics`.

        Report-entry and formatter metadata is not included; use
        :meth:`report` or :meth:`write_json_string` for those.

        Disabled statistics always return an empty dict.

        Returns
        -------
        A ``dict`` mapping each stat name to its ``{"count", "value", "max"}`` dict.
        """
        return json.loads(self.write_json_string())["statistics"]

    def add_stat(self, name, double value):
        """
        Adds a value to a statistic.

        Parameters
        ----------
        name
            Name of the statistic.
        value
            Value to add.
        """
        cdef string name_ = str.encode(name)
        with nogil:
            deref(self._handle).add_stat(name_, value)

    def add_report_entry(self, name, stat_names, Formatter formatter):
        """
        Associate a predefined formatter with one or more stat names.

        Mirrors the C++ ``rapidsmpf::Statistics::add_report_entry``.
        First-wins: if a report entry already exists under ``name``, this
        call has no effect.

        Parameters
        ----------
        name
            Report entry name. Becomes one line in :meth:`report`.
        stat_names
            Iterable of stat names this entry aggregates. The number of
            names must match the arity of ``formatter``.
        formatter
            A `Formatter` selecting the predefined render function.
        """
        cdef string name_ = str.encode(name)
        cdef vector[string] cpp_stat_names
        for sn in stat_names:
            cpp_stat_names.push_back(str.encode(sn))
        with nogil:
            deref(self._handle).add_report_entry(
                name_, cpp_stat_names, formatter
            )

    def copy(self):
        """
        Creates a deep copy of this Statistics object.

        Memory records are not copied.

        Returns
        -------
        A new Statistics with the same stats and formatters.
        """
        cdef Statistics ret = Statistics.__new__(Statistics)
        with nogil:
            ret._handle = deref(self._handle).copy()
        return ret

    @staticmethod
    def merge(stats):
        """
        Merge a sequence of Statistics into a new one.

        For each stat name present in any input, the result has the summed
        count, summed value, and the maximum of the maxes. Report entries
        with the same name must agree on formatter and stat-name list;
        otherwise the call raises ``ValueError``. Memory records are not
        merged.

        Parameters
        ----------
        stats
            A non-empty sequence of :class:`Statistics` to merge.

        Returns
        -------
        A new :class:`Statistics` containing the merged data.

        Raises
        ------
        ValueError
            If ``stats`` is empty or two inputs have conflicting report
            entries.
        """
        cdef Statistics ret = Statistics.__new__(Statistics)
        cdef vector[shared_ptr[cpp_Statistics]] v
        for item in stats:
            v.push_back((<Statistics?>item)._handle)
        with nogil:
            ret._handle = cpp_merge_statistics(v)
        return ret

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

    def memory_profiling(self, RmmResourceAdaptor mr, name):
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

        Pass ``mr=None`` to get a no-op recorder.

        Parameters
        ----------
        mr
            The memory resource through which allocations are tracked.
            Pass ``None`` to get a no-op recorder.
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
        >>> stats = Statistics(enable=True)
        >>> with stats.memory_profiling(mr, "outer"):
        ...     b1 = rmm.DeviceBuffer(size=1024, mr=mr)
        ...     with stats.memory_profiling(mr, "inner"):
        ...         b2 = rmm.DeviceBuffer(size=1024, mr=mr)
        >>> inner = stats.get_memory_records()["inner"]
        >>> print(inner.scoped.peak())
        1024
        >>> outer = stats.get_memory_records()["outer"]
        >>> print(outer.scoped.peak())
        2048
        """
        return MemoryRecorder(self, mr, name)

    def clear(self) -> None:
        """
        Clears all statistics.

        Memory profiling records are not cleared.
        """
        with nogil:
            cpp_clear_statistics(deref(self._handle))

    def write_json(self, filepath) -> None:
        """
        Writes a JSON report of all collected statistics to a file.

        Disabled statistics produce a JSON object with an empty ``statistics``
        section.

        Parameters
        ----------
        filepath
            Path to the output file. Created or overwritten.

        Raises
        ------
        OSError
            If the file cannot be opened or writing fails.
        """
        cdef string path = <bytes>os.fsencode(filepath)
        with nogil:
            cpp_write_json(deref(self._handle), path)

    def copy(self):
        """
        Creates a deep copy of this Statistics object.

        Memory records are not copied.

        Returns
        -------
        A new Statistics with the same stats and formatters.
        """
        cdef Statistics ret = Statistics.__new__(Statistics)
        with nogil:
            ret._handle = deref(self._handle).copy()
        return ret

    @staticmethod
    def merge(stats):
        """
        Merge a sequence of Statistics into a new one.

        For each stat name present in any input, the result has the summed
        count, summed value, and the maximum of the maxes. Report entries
        with the same name must agree on formatter and stat-name list;
        otherwise the call raises ``ValueError``. Memory records are not
        merged.

        Parameters
        ----------
        stats
            A non-empty sequence of :class:`Statistics` to merge.

        Returns
        -------
        A new :class:`Statistics` containing the merged data.

        Raises
        ------
        ValueError
            If ``stats`` is empty or two inputs have conflicting report
            entries.
        """
        cdef Statistics ret = Statistics.__new__(Statistics)
        cdef vector[shared_ptr[cpp_Statistics]] v
        for item in stats:
            v.push_back((<Statistics?>item)._handle)
        with nogil:
            ret._handle = cpp_merge_statistics(v)
        return ret

    def write_json_string(self) -> str:
        """
        Returns a JSON representation of all collected statistics as a string.

        Disabled statistics produce a JSON object with an empty ``statistics``
        section.

        Returns
        -------
        A JSON-formatted string.
        """
        cdef string result
        with nogil:
            result = cpp_write_json_string(deref(self._handle))
        return result.decode("utf-8")

    def serialize(self) -> bytes:
        """
        Serialize the stats and report entries to a binary buffer.

        Memory records and the memory-profiling resource pointer are not
        included.

        Returns
        -------
        A ``bytes`` object containing the serialized binary representation
        of the Statistics.
        """
        cdef vector[uint8_t] vec
        with nogil:
            vec = deref(self._handle).serialize()
        return <bytes>PyBytes_FromStringAndSize(
            <const char*>vec.data() if not vec.empty() else NULL,
            vec.size()
        )

    @staticmethod
    def deserialize(bytes buf not None):
        """
        Deserialize a binary buffer into a Statistics object.

        Reconstructs a Statistics instance from a byte buffer produced by
        :meth:`serialize`. The resulting object has no memory records and
        no associated memory-profiling resource.

        Parameters
        ----------
        buf
            A buffer containing serialized statistics.

        Returns
        -------
        A reconstructed :class:`Statistics` instance.

        Raises
        ------
        ValueError
            If the input buffer is malformed or truncated.
        """
        cdef Py_ssize_t size = len(buf)
        cdef const char* src = <const char*>buf
        cdef vector[uint8_t] vec
        cdef Statistics ret = Statistics.__new__(Statistics)
        with nogil:
            vec.resize(size)
            memcpy(<void*>vec.data(), src, size)
            ret._handle = cpp_deserialize_statistics(vec)
        return ret

    def __getstate__(self):
        return self.serialize()

    def __setstate__(self, bytes state not None):
        cdef Statistics restored = Statistics.deserialize(state)
        self._handle = restored._handle


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
                self._stats._handle.get(), deref(mr), self._name
            )

    def __exit__(self, exc_type, exc_value, traceback):
        if self._mr is not None:
            with nogil:
                self._handle.reset()
        return False  # do not suppress exceptions
