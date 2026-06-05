# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""The ProgressThread interface for RapidsMPF."""

from libcpp.memory cimport make_shared

from rapidsmpf.runtime cimport Runtime


cdef class ProgressThread:
    """
    A progress thread that can execute arbitrary functions.

    The ``ProgressThread`` class provides an interface for executing arbitrary
    functions in a separate thread. The functions are executed in the order they
    were registered, and a newly registered function will only execute for the
    first time in the next iteration of the progress thread.

    Parameters
    ----------
    runtime
        The Runtime context providing statistics and configuration.

    Notes
    -----
    This class is designed to handle background tasks and progress tracking in
    distributed operations. It is typically used in conjunction with the
    ``Shuffler`` class to track progress of data movement operations.
    """
    def __init__(self, Runtime runtime not None):
        with nogil:
            self._handle = make_shared[cpp_ProgressThread](runtime._handle)

    def __dealloc__(self):
        with nogil:
            self._handle.reset()
