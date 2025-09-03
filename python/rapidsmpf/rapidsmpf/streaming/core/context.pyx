# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from rapidsmpf.buffer.resource cimport BufferResource, cpp_BufferResource
from rapidsmpf.communicator.communicator cimport Communicator
from rapidsmpf.config cimport Options
from rapidsmpf.statistics cimport Statistics

from rapidsmpf.config import get_environment_variables

from libcpp.memory cimport make_shared


cdef class Context:
    """
    Context for streaming nodes (coroutines) in RapidsMPF.

    Parameters
    ----------
    comm
        The communicator to use.
    br
        Buffer resource to use.
    options
        Configuration options to use. Missing config options are read
        from environment variables.
    statistics
        The statistics instance to use. If None, statistics are disabled.
    """
    def __cinit__(
        self,
        Communicator comm,
        BufferResource br,
        Options options = None,
        Statistics statistics = None,
    ):
        self._comm = comm
        self._br = br
        cdef cpp_BufferResource* _br = br.ptr()

        self._options = options
        if self._options is None:
            self._options = Options()
        # Insert missing config options from environment variables.
        self._options.insert_if_absent(get_environment_variables())

        self._statistics = statistics
        if statistics is None:
            self._statistics = Statistics(enable=False)  # Disables statistics.

        with nogil:
            self._handle = make_shared[cpp_Context](
                self._options._handle,
                self._comm._handle,
                _br,
                self._statistics._handle,
            )

    def __dealloc__(self):
        with nogil:
            self._handle.reset()

    def options(self):
        return self._options

    def comm(self):
        return self._options

    def br(self):
        return self._br

    def statistics(self):
        return self._statistics
