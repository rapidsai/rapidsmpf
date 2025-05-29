# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference as deref
from libcpp cimport bool
from libcpp.cast cimport dynamic_cast
from libcpp.memory cimport make_shared
from libcpp.string cimport string
from rmm.librmm.memory_resource cimport (device_memory_resource,
                                         statistics_resource_adaptor)


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

# Alias of a `rmm::mr::statistics_resource_adaptor` pointer.
ctypedef statistics_resource_adaptor[device_memory_resource]* stats_mr_ptr

cdef class Statistics:
    """
    Track statistics across RapidsMPF operations.

    Parameters
    ----------
    enable
        Whether statistics tracking is enabled.
    mr
        The statistics memory resource used for memory profiling. If None,
        memory profiling is disabled.
    """
    def __cinit__(self, bool enable, StatisticsResourceAdaptor mr = None):
        self._mr = mr
        if self._mr is None:
            with nogil:
                self._handle = make_shared[cpp_Statistics](enable)
            return
        cdef stats_mr_ptr m = dynamic_cast[stats_mr_ptr](self._mr.get_mr())
        assert m  # The dynamic cast should always succeed.
        with nogil:
            self._handle = make_shared[cpp_Statistics](enable, m)

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
        True if the object is enabled, otherwise false.
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


cdef shared_ptr[cpp_Statistics] parse_statistic_argument(
    Statistics stats
) noexcept nogil:
    """
    Convert a Python `Statistics` object its C++ representation.

    This helper is used to extract the underlying `std::shared_ptr<Statistics>` used
    by C++ APIs. If the input `stats` is `None` or has no memory recorder, the globally
    disabled statistics object is returned.

    Parameters
    ----------
    stats
        A Python wrapper around the C++ Statistics object. Can be `None`.

    Returns
    -------
    A shared pointer to the underlying C++ Statistics instance. Returns a disabled
    Statistics instance if `stats` is `None` or not initialized.
    """
    if stats is None or stats._mr is None:
        return cpp_Statistics.disabled()
    assert stats._handle
    return stats._handle
