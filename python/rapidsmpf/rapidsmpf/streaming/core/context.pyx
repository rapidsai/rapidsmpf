# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference as deref
from libcpp.utility cimport move

from rapidsmpf.communicator.communicator cimport Communicator
from rapidsmpf.config cimport Options
from rapidsmpf.memory.buffer_resource cimport BufferResource

from rapidsmpf.config import get_environment_variables

from libcpp.memory cimport make_shared
from rmm.pylibrmm.stream cimport Stream

from rapidsmpf.rmm_resource_adaptor cimport RmmResourceAdaptor
from rapidsmpf.streaming.core.channel cimport Channel, cpp_Channel
from rapidsmpf.streaming.core.memory_reserve_or_wait cimport \
    MemoryReserveOrWait

from rapidsmpf.memory.buffer import MemoryType as py_MemoryType


cdef class Context:
    """
    Context for actors (coroutines) in rapidsmpf.

    The context owns shared resources used during execution, including the
    coroutine executor and memory reservation infrastructure.

    A ``Context`` instance must be created and shut down on the same thread.
    Shutting down the context from a different thread results in program
    termination. This is particularly important in coroutine-based code, where
    execution and stack unwinding may occur on different threads if ownership
    is not carefully managed.

    In Python, it is easy to accidentally keep dangling references to a
    ``Context`` instance, which may delay destruction and cause shutdown to
    occur on an unintended thread. For this reason, it is strongly recommended
    to use ``Context`` as a context manager (that is, via a ``with`` statement),
    which guarantees that ``shutdown()`` is invoked deterministically and on
    the same thread that created the context.

    Parameters
    ----------
    comm
        The communicator to use.
    br
        The buffer resource to use.
    options
        The configuration options to use. Missing options are read from environment
        variables.

    Examples
    --------
    >>> with streaming.Context(
    ...     comm=...,
    ...     br=BufferResource(...),
    ...     options=Options(...),
    ... ) as ctx:
    ...     ch = ctx.create_channel()
    """
    def __cinit__(
        self,
        Communicator comm not None,
        BufferResource br not None,
        Options options = None,
    ):
        self._comm = comm
        self._br = br

        self._options = options
        if self._options is None:
            self._options = Options()
        # Insert missing config options from environment variables.
        self._options.insert_if_absent(get_environment_variables())

        with nogil:
            self._handle = make_shared[cpp_Context](
                self._options._handle,
                self._comm._handle,
                self._br._handle,
            )

        self._spillable_messages = SpillableMessages.from_handle(
            deref(self._handle).spillable_messages()
        )
        self._memory = {}
        for mem_type in py_MemoryType:
            self._memory[mem_type] = MemoryReserveOrWait.from_handle(
                deref(self._handle).memory(mem_type), self._br
            )

    @classmethod
    def from_options(
        cls,
        Communicator comm not None,
        RmmResourceAdaptor mr not None,
        Options options not None
    ):
        return cls(
            comm=comm,
            br=BufferResource.from_options(mr, options),
            options=options,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown()
        return False  # do not suppress exceptions

    def __dealloc__(self):
        # Shut down the C++ context explicitly to ensure shutdown happens immediately
        # and not later via a dangling reference on another thread. Recall that
        # shutting down a C++ context on a different thread than the one that created
        # it results in program termination.
        with nogil:
            deref(self._handle).shutdown()
            self._handle.reset()

    def shutdown(self):
        """
        Shut down the context.

        This method is idempotent and only performs shutdown once. Subsequent calls
        have no effect.

        Warnings
        --------
        Shutdown must be initiated from the same thread that constructed the
        executor. Calling this method from a different thread results in program
        termination.
        """
        with nogil:
            deref(self._handle).shutdown()

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
        return self._br.statistics

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
        return self._br.stream_pool_size()

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

    def spillable_messages(self):
        """
        Get spillable messages.

        Returns
        -------
        The spillable messages associated with this context.
        """
        return self._spillable_messages

    def memory(self, MemoryType mem_type):
        """
        Get the memory reservation handle for a given memory type.

        Returns an object that coordinates asynchronous memory reservation requests
        for the specified memory type. The returned instance provides backpressure
        and global progress guarantees and should be used to reserve memory before
        performing operations that require memory.

        A recommended usage pattern is to reserve all required memory up front as a
        single atomic reservation. This allows callers to await the reservation and
        only start executing the operation once all required memory is available.

        Parameters
        ----------
        mem_type
            Memory type for which reservations are requested.

        Returns
        -------
        Handle that coordinates memory reservation requests for the given memory type.
        """
        return self._memory[mem_type]
