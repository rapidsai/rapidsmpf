# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference as deref
from libcpp cimport bool
from libcpp.memory cimport make_shared


cdef class Statistics:
    """
    Track statistics across rapidsmp operations.

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
        return deref(self._handle).report().decode('UTF-8')
