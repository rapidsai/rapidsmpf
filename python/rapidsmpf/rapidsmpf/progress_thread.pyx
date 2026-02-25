# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""The ProgressThread interface for RapidsMPF."""

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
        The statistics instance to use. If None, statistics is disabled.

    Notes
    -----
    This class is designed to handle background tasks and progress tracking in
    distributed operations. It is typically used in conjunction with the
    `Shuffler` class to track progress of data movement operations.
    """
    def __init__(
        self,
        Statistics statistics = None,
    ):
        if statistics is None:
            statistics = Statistics(enable=False)  # Disables statistics.

        with nogil:
            self._handle = make_shared[cpp_ProgressThread](statistics._handle)

    def __dealloc__(self):
        with nogil:
            self._handle.reset()
