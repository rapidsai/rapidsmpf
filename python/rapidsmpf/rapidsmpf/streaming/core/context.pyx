# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from rapidsmpf.buffer.resource cimport BufferResource, cpp_BufferResource
from rapidsmpf.communicator.communicator cimport Communicator
from rapidsmpf.config cimport Options
from rapidsmpf.statistics cimport Statistics

from rapidsmpf.config import get_environment_variables

from libcpp.memory cimport make_shared


cdef class Context:
    def __cinit__(
        self,
        Communicator comm,
        BufferResource br,
        Options options = None,
        Statistics statistics = None,
    ):
        self._comm = comm
        self._br = br
        cdef cpp_BufferResource* br_ = br.ptr()

        if options is None:
            options = Options()
        # Insert missing config options from environment variables.
        options.insert_if_absent(get_environment_variables())

        if statistics is None:
            statistics = Statistics(enable=False)  # Disables statistics.

        with nogil:
            self._handle = make_shared[cpp_Context](
                options._handle,
                self._comm._handle,
                br_,
                statistics._handle,
            )

    def __dealloc__(self):
        with nogil:
            self._handle.reset()
