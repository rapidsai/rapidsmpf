# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference as deref
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.utility cimport move

from rapidsmpf.progress_thread cimport ProgressThread


cdef class Logger:
    """
    Logger.

    To control the verbosity level, set the environment variable ``RAPIDSMPF_LOG``:
      - NONE:  No logging.
      - PRINT: General print messages.
      - WARN:  Warning messages (default)
      - INFO:  Informational messages.
      - DEBUG: Debug messages.
      - TRACE: Trace messages.
    """

    def __init__(self):
        raise TypeError("Please get a `Logger` from a communicator instance")

    def __dealloc__(self):
        with nogil:
            self._handle.reset()

    @property
    def verbosity_level(self):
        """
        Get the verbosity level of the logger.

        Returns
        -------
            The verbosity level.
        """
        return deref(self._handle).verbosity_level()

    def print(self, str msg not None):
        """
        Logs a print message.

        Parameters
        ----------
        msg
            The message to log.
        """
        cdef string _msg = msg.encode()
        with nogil:
            deref(self._handle).log(LOG_LEVEL.PRINT, move(_msg))

    def warn(self, str msg not None):
        """
        Logs a warning message.

        Parameters
        ----------
        msg
            The message to log.
        """
        cdef string _msg = msg.encode()
        with nogil:
            deref(self._handle).log(LOG_LEVEL.WARN, move(_msg))

    def info(self, str msg not None):
        """
        Logs an informational message.

        Parameters
        ----------
        msg
            The message to log.
        """
        cdef string _msg = msg.encode()
        with nogil:
            deref(self._handle).log(LOG_LEVEL.INFO, move(_msg))

    def debug(self, str msg not None):
        """
        Logs a debug message.

        Parameters
        ----------
        msg
            The message to log.
        """
        cdef string _msg = msg.encode()
        with nogil:
            deref(self._handle).log(LOG_LEVEL.DEBUG, move(_msg))

    def trace(self, str msg not None):
        """
        Logs a trace message.

        Parameters
        ----------
        msg
            The message to log.
        """
        cdef string _msg = msg.encode()
        with nogil:
            deref(self._handle).log(LOG_LEVEL.TRACE, move(_msg))


cdef class Communicator:
    """
    Abstract base class for a communication mechanism between nodes.

    Provides an interface for sending and receiving messages between nodes,
    supporting asynchronous operations, GPU data transfers, and custom logging.
    Concrete implementations must define the virtual methods to enable specific
    communication backends.

    Notes
    -----
    This class is designed as an abstract base class, meaning it cannot be
    instantiated directly. Subclasses are required to implement the necessary
    methods to support the desired communication backend and functionality.
    """
    def __init__(self):
        raise TypeError(
            "Communicator is an abstract base case, please create a "
            "communicator through a concrete implementation such as "
            "`rapidsmpf.mpi.new_communicator()`"
        )

    def __dealloc__(self):
        with nogil:
            self._handle.reset()

    @property
    def rank(self):
        """
        Get the rank of this communication node.

        Returns
        -------
            The rank.
        """
        return deref(self._handle).rank()

    @property
    def nranks(self):
        """
        Get the total number of ranks.

        Returns
        -------
            Total number of ranks.
        """
        return deref(self._handle).nranks()

    @property
    def logger(self):
        """
        Get the logger.

        Returns
        -------
            A logger instance.
        """
        cdef Logger logger = Logger.__new__(Logger)
        logger._handle = deref(self._handle).logger()
        return logger

    @property
    def progress_thread(self):
        """
        Get the communicator's progress thread.

        Returns
        -------
        The progress thread.
        """
        cdef ProgressThread pt = ProgressThread.__new__(ProgressThread)
        pt._handle = deref(self._handle).progress_thread()
        return pt

    def get_str(self):
        """
        Get a string representation of the communicator.

        Returns
        -------
            A string describing the communicator
        """
        cdef string s = deref(self._handle).str()
        return s.decode('utf-8')

    @property
    def aborted(self):
        """
        Whether the communicator has been aborted, by this or another rank.

        A communicator is aborted by :meth:`abort`, or when an operation is
        destroyed while its communication is in flight, e.g., because the
        operation failed. Its communication then raises
        :class:`~rapidsmpf.error.CommunicatorAborted`.

        Returns
        -------
            True if the communicator has been aborted.
        """
        return deref(self._handle).is_aborted()

    def abort(self, str reason not None, *, bint terminate = False):
        """
        Abort the communicator on all ranks.

        Fail-safe for unrecoverable errors. When an operation fails on one rank,
        for example a collective that raised an exception, the other ranks may
        wait forever for messages that will never arrive. Aborting marks the
        communicator as aborted on this rank and notifies the other ranks, best
        effort, which mark it as aborted too. They are also notified if this
        rank already aborted the communicator by destroying an operation with
        communication in flight, which is the usual case after a failed
        operation. From then on, communication through the communicator raises
        :class:`~rapidsmpf.error.CommunicatorAborted`, including from the
        operations waiting on it, such as ``Shuffler.wait()`` or
        ``AllGather.wait_and_extract()``. Every rank thus gets an exception it
        can handle, e.g., by unwinding and exiting.

        The other ranks are notified by:

        - MPI: a message on a reserved tag, received when polling for messages.
        - UCXX: a control message, received by the UCX worker.
        - Single: there are no other ranks.

        In-flight communication is not cancelled; an aborted communicator must
        not be used for communication anymore.

        RapidsMPF never calls this itself: errors are reported as exceptions,
        and it is up to the application to decide whether an error is
        recoverable or whether the communicator must be aborted.

        Parameters
        ----------
        reason
            Human-readable reason, forwarded to the other ranks and included in
            the message of the exceptions, best effort: each rank only records
            the first reason it receives. If the communicator was already
            aborted, e.g., by destroying an operation with communication in
            flight or by another rank, the exceptions on this rank carry that
            earlier reason, and so may those on the other ranks, e.g., if an
            operation running on the progress thread notified them first. The
            reason is always printed to ``stderr``.
        terminate
            If True, terminate the processes of all ranks instead, without
            unwinding: ``MPI_Abort()`` with MPI, and every rank terminating upon
            receiving the abort message with UCXX. The other ranks are not
            notified cooperatively first, so that they don't start unwinding.
            This is a last resort, e.g., when other ranks may not be making
            progress. Does not return.
        """
        cdef string _reason = reason.encode()
        cdef bool _terminate = terminate
        with nogil:
            deref(self._handle).abort(_reason, _terminate)


def _available_communicators():
    ret = ["single"]
    if COMM_HAVE_UCXX:
        ret.append("ucxx")
    if COMM_HAVE_MPI:
        ret.append("mpi")
    return tuple(ret)
