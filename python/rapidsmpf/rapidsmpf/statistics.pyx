# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference as deref
from libcpp cimport bool
from libcpp.memory cimport make_shared
from libcpp.string cimport string


# Since `Statistics::Stat` doesn't have a default ctor, we use the following
# getters.
cdef extern from *:
    """
    std::size_t cpp_get_statistic_count(
        rapidsmp::Statistics const& stats, std::string const& name
    ) {
        return stats.get_stat(name).count();
    }
    std::size_t cpp_get_statistic_value(
        rapidsmp::Statistics const& stats, std::string const& name
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
    """
    def __cinit__(self, bool enable):
        with nogil:
            self._handle = make_shared[cpp_Statistics](enable)

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
