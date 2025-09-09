# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typing

from cython.operator cimport dereference as deref
from cython.operator cimport postincrement
from libcpp cimport bool as cpp_bool
from libcpp.memory cimport make_unique
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from rapidsmpf.cupti cimport milliseconds


cdef class MemoryDataPoint:
    """A data point representing GPU memory usage at a specific time.

    Attributes
    ----------
    timestamp
        Time when sample was taken (seconds since epoch)
    free_memory
        Free GPU memory in bytes
    total_memory
        Total GPU memory in bytes
    used_memory
        Used GPU memory in bytes
    """

    def __init__(self):
        raise ValueError("Use the `from_cpp` factory method")

    @staticmethod
    cdef MemoryDataPoint from_cpp(cpp_MemoryDataPoint data):
        """Create a MemoryDataPoint from C++ data."""
        cdef MemoryDataPoint result = MemoryDataPoint.__new__(MemoryDataPoint)
        result._data = data
        return result

    @property
    def timestamp(self) -> float:
        """Time when sample was taken (seconds since epoch)."""
        return self._data.timestamp

    @property
    def free_memory(self) -> int:
        """Free GPU memory in bytes."""
        return self._data.free_memory

    @property
    def total_memory(self) -> int:
        """Total GPU memory in bytes."""
        return self._data.total_memory

    @property
    def used_memory(self) -> int:
        """Used GPU memory in bytes."""
        return self._data.used_memory

    def __repr__(self) -> str:
        return (f"MemoryDataPoint(timestamp={self.timestamp}, "
                f"free_memory={self.free_memory}, "
                f"total_memory={self.total_memory}, "
                f"used_memory={self.used_memory})")


cdef class CuptiMonitor:
    """CUDA memory monitoring using CUPTI (CUDA Profiling Tools Interface).

    This class provides memory monitoring capabilities for CUDA applications
    by intercepting CUDA runtime and driver API calls related to memory
    operations and kernel launches.
    """

    def __cinit__(self, enable_periodic_sampling=False, sampling_interval_ms=100):
        """Initialize a CuptiMonitor instance.

        Parameters
        ----------
        enable_periodic_sampling
            Enable background thread for periodic memory sampling
        sampling_interval_ms
            Interval between periodic samples in milliseconds
        """
        cdef cpp_bool enable_sampling = <cpp_bool>enable_periodic_sampling

        self._handle = make_unique[cpp_CuptiMonitor](
            enable_sampling,
            milliseconds(<long long>sampling_interval_ms),
        )

    def __dealloc__(self):
        """Destructor - automatically stops monitoring and cleans up CUPTI."""
        with nogil:
            self._handle.reset()

    def start_monitoring(self) -> None:
        """Start memory monitoring.

        Initializes CUPTI and begins intercepting CUDA API calls.

        Raises
        ------
        RuntimeError
            If CUPTI initialization fails
        """
        with nogil:
            self._handle.get().start_monitoring()

    def stop_monitoring(self) -> None:
        """Stop memory monitoring.

        Stops CUPTI callbacks and periodic sampling if enabled.
        """
        with nogil:
            self._handle.get().stop_monitoring()

    def is_monitoring(self) -> bool:
        """Check if monitoring is currently active.

        Returns
        -------
        True if monitoring is active, False otherwise
        """
        cdef cpp_bool result
        with nogil:
            result = self._handle.get().is_monitoring()
        return result

    def capture_memory_sample(self) -> None:
        """Manually capture current memory usage.

        This can be called at any time to manually record a memory sample,
        regardless of whether periodic sampling is enabled.
        """
        with nogil:
            self._handle.get().capture_memory_sample()

    def get_memory_samples(self) -> typing.List[MemoryDataPoint]:
        """Get all collected memory samples.

        Returns
        -------
        List of memory data points
        """
        cdef const vector[cpp_MemoryDataPoint]* samples
        with nogil:
            samples = &self._handle.get().get_memory_samples()

        cdef list result = []
        cdef size_t i
        for i in range(samples.size()):
            result.append(MemoryDataPoint.from_cpp(deref(samples)[i]))
        return result

    def clear_samples(self) -> None:
        """Clear all collected memory samples."""
        with nogil:
            self._handle.get().clear_samples()

    def get_sample_count(self) -> int:
        """Get the number of memory samples collected.

        Returns
        -------
        Number of samples
        """
        cdef size_t count
        with nogil:
            count = self._handle.get().get_sample_count()
        return <int>count

    def write_csv(self, filename: str) -> None:
        """Write memory samples to CSV file.

        Parameters
        ----------
        filename
            Output CSV filename

        Raises
        ------
        RuntimeError
            If file cannot be written
        """
        cdef string c_filename = filename.encode('utf-8')
        with nogil:
            self._handle.get().write_csv(c_filename)

    def set_debug_output(self, enabled: bool, threshold_mb: int = 10) -> None:
        """Enable or disable debug output for significant memory changes.

        Parameters
        ----------
        enabled
            If True, prints debug info when memory usage changes significantly
        threshold_mb
            Threshold in MB for what constitutes a "significant" change (default: 10)
        """
        cdef cpp_bool c_enabled = <cpp_bool>enabled
        cdef size_t c_threshold = <size_t>threshold_mb
        with nogil:
            self._handle.get().set_debug_output(c_enabled, c_threshold)

    def get_callback_counters(self) -> typing.Dict[int, int]:
        """Get callback counters for all monitored CUPTI callbacks.

        Returns a dictionary where keys are CUPTI callback IDs and values are the number
        of times each callback was triggered during monitoring.

        Returns
        -------
        Dictionary from CUPTI callback ID to call count
        """
        cdef unordered_map[CUpti_CallbackId, size_t] counters
        with nogil:
            counters = self._handle.get().get_callback_counters()

        cdef dict result = {}
        cdef unordered_map[CUpti_CallbackId, size_t].iterator it = counters.begin()
        while it != counters.end():
            result[<int>deref(it).first] = <int>deref(it).second
            postincrement(it)
        return result

    def clear_callback_counters(self) -> None:
        """Clear all callback counters.

        Resets all callback counters to zero.
        """
        with nogil:
            self._handle.get().clear_callback_counters()

    def get_total_callback_count(self) -> int:
        """Get total number of callbacks triggered across all monitored callback IDs.

        Returns
        -------
        Total number of callbacks
        """
        cdef size_t count
        with nogil:
            count = self._handle.get().get_total_callback_count()
        return <int>count

    def get_callback_summary(self) -> str:
        """Get a human-readable summary of callback counters.

        Returns a formatted string showing callback names and their counts.

        Returns
        -------
        String containing callback counter summary
        """
        cdef string summary
        with nogil:
            summary = self._handle.get().get_callback_summary()
        return summary.decode('utf-8')
