# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from rmm.librmm.cuda_stream_pool cimport cuda_stream_pool

from rapidsmpf.buffer.resource cimport BufferResource, cpp_BufferResource
from rapidsmpf.communicator.communicator cimport Communicator
from rapidsmpf.config cimport Options
from rapidsmpf.statistics cimport Statistics

from rapidsmpf.config import get_environment_variables

from libc.stddef cimport size_t
from libcpp.memory cimport make_shared
from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.pylibrmm.stream cimport Stream


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
        return self._comm

    def br(self):
        return self._br

    def statistics(self):
        return self._statistics

    def get_stream_from_pool(self) -> Stream:
        """
        Get a stream from the stream pool.

        Returns
        -------
        Stream
            A stream from the stream pool.
        """
        cdef const cuda_stream_pool* pool_ptr = self._br.stream_pool()
        cdef cuda_stream_view stream_view
        with nogil:
            stream_view = pool_ptr.get_stream()
        # passing the buffer resource as the owner of the stream so that it is kept
        # alive for the lifetime of the Stream obj
        return Stream._from_cudaStream_t(stream_view.value(), self._br)

    def stream_pool_size(self) -> int:
        """
        Get the size of the stream pool.

        Returns
        -------
        int
            The size of the stream pool.
        """
        cdef const cuda_stream_pool* pool_ptr = self._br.stream_pool()
        cdef size_t size
        with nogil:
            size = pool_ptr.get_pool_size()
        return int(size)
