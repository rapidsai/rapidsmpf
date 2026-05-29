# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""The ProgressThread interface for RapidsMPF."""

from cython.operator cimport dereference as deref
from libcpp.memory cimport make_shared

from rapidsmpf.statistics cimport Statistics


cdef class ProgressThread:
    """
    A progress thread that can execute arbitrary functions.

    The `ProgressThread` class provides an interface for executing arbitrary
    functions in a separate thread. The functions are executed in the order they
    were registered, and a newly registered function will only execute for the
    first time in the next iteration of the progress thread.

    Parameters
    ----------
    statistics
        The statistics instance to use. Required. Pass
        ``Statistics.disabled()`` for a no-op recorder.

    Notes
    -----
    This class is designed to handle background tasks and progress tracking in
    distributed operations. It is typically used in conjunction with the
    `Shuffler` class to track progress of data movement operations.
    """
    def __init__(
        self,
        Statistics statistics not None,
    ):
        with nogil:
            self._handle = make_shared[cpp_ProgressThread](statistics._handle)

    @property
    def statistics(self):
        """
        Get the statistics instance associated with this progress thread.

        Returns
        -------
            The Statistics instance.
        """
        cdef Statistics stats = Statistics.__new__(Statistics)
        stats._handle = deref(self._handle).statistics()
        return stats

    def set_statistics(self, Statistics statistics not None):
        """
        Replace the statistics instance held by this progress thread.

        The swap is performed atomically.

        Parameters
        ----------
        statistics
            The new statistics instance. Pass ``Statistics.disabled()`` to opt out of
            statistics collection.
        """
        with nogil:
            deref(self._handle).set_statistics(statistics._handle)

    def __dealloc__(self):
        with nogil:
            self._handle.reset()
