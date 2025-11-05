# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference as deref
from libcpp.utility cimport move

from rapidsmpf.buffer.resource cimport BufferResource, cpp_BufferResource
from rapidsmpf.communicator.communicator cimport Communicator
from rapidsmpf.config cimport Options
from rapidsmpf.statistics cimport Statistics

from rapidsmpf.config import get_environment_variables

from libcpp.memory cimport make_shared
from rmm.pylibrmm.stream cimport Stream

from rapidsmpf.streaming.core.channel cimport Channel, cpp_Channel


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
        """
        Get options.

        Returns
        -------
        The options associated with this context.
        """
        return self._options

    def comm(self):
        """
        Get the communicator.

        Returns
        -------
        The communicator associated with this context.
        """
        return self._comm

    def br(self):
        """
        Get buffer resource.

        Returns
        -------
        The buffer resource associated with this context.
        """
        return self._br

    def statistics(self):
        """
        Get statistics.

        Returns
        -------
        The statistics associated with this context.
        """
        return self._statistics

    def get_stream_from_pool(self):
        """
        Get a stream from the stream pool.

        Returns
        -------
        A stream from the stream pool.
        """
        # passing the buffer resource as the owner of the stream so that it is kept
        # alive for the lifetime of the Stream obj
        return Stream._from_cudaStream_t(
            self._br.stream_pool().get_stream().value(), self._br
        )

    def stream_pool_size(self):
        """
        Get the size of the stream pool.

        Returns
        -------
        The size of the stream pool.
        """
        return self._br.stream_pool().get_pool_size()

    def create_channel(self):
        """
        Create a new channel associated with this context.

        Returns
        -------
        The newly created channel.
        """
        cdef shared_ptr[cpp_Channel] ret
        with nogil:
            ret = deref(self._handle).create_channel()
        return Channel.from_handle(move(ret))
