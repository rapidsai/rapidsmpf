# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""The AllGather interface for RapidsMPF."""

from cython.operator cimport dereference as deref
from libc.stdint cimport uint8_t, uint64_t
from libcpp.memory cimport make_unique
from libcpp.utility cimport move
from libcpp.vector cimport vector

from rapidsmpf.allgather.allgather cimport (Ordered, cpp_AllGather,
                                            milliseconds_t)
from rapidsmpf.communicator.communicator cimport Communicator
from rapidsmpf.memory.buffer_resource cimport (BufferResource,
                                               cpp_BufferResource)
from rapidsmpf.memory.packed_data cimport (PackedData, cpp_PackedData,
                                           packed_data_vector_to_list)
from rapidsmpf.progress_thread cimport ProgressThread
from rapidsmpf.statistics cimport Statistics


cdef class AllGather:
    """
    AllGather communication service for distributed operations.

    The `rapidsmpf.allgather.AllGather` class provides a communication service
    where each rank contributes data and all ranks receive all inputs from all ranks.

    The implementation uses a ring broadcast algorithm where each rank receives a
    contribution from its left neighbor, forwards the message to its right neighbor
    (unless at the end of the ring) and then stores the contribution locally.

    Parameters
    ----------
    comm
        The communicator for communication between ranks.
    progress_thread
        The progress thread for asynchronous operations.
    op_id
        Unique operation identifier for this allgather. Must have a value
        between 0 and 255.
    br
        Buffer resource for memory allocation.
    statistics
        Statistics collection instance. If None, statistics is disabled.

    Notes
    -----
    The caller promises that inserted buffers are stream-ordered with respect to
    their own stream, and extracted buffers are likewise guaranteed to be stream-
    ordered with respect to their own stream.
    """

    def __init__(
        self,
        Communicator comm not None,
        ProgressThread progress_thread not None,
        uint8_t op_id,
        BufferResource br not None,
        Statistics statistics = None,
    ):
        self._comm = comm
        self._br = br
        cdef cpp_BufferResource* br_ = br.ptr()
        if statistics is None:
            statistics = Statistics(enable=False)  # Disables statistics.
        with nogil:
            self._handle = make_unique[cpp_AllGather](
                comm._handle,
                progress_thread._handle,
                op_id,
                br_,
                statistics._handle,
            )

    def __dealloc__(self):
        with nogil:
            self._handle.reset()

    @property
    def comm(self):
        """
        Get the communicator used by the allgather.

        Returns
        -------
        The communicator.
        """
        return self._comm

    def insert(self, uint64_t sequence_number, PackedData packed_data):
        """
        Insert packed data into the allgather operation (non-blocking).

        Parameters
        ----------
        sequence_number
            The sequence number of this insertion, used when extracting ordered data.
        packed_data
            The data to contribute to the allgather.

        Notes
        -----
        This method adds the given data to the allgather operation. The data
        will be distributed to all participating ranks.
        """
        if not packed_data.c_obj:
            raise ValueError("PackedData was empty")

        with nogil:
            deref(self._handle).insert(sequence_number, move(deref(packed_data.c_obj)))

    def insert_finished(self):
        """
        Mark that this rank has finished contributing data.

        Notes
        -----
        This method signals that no more data will be inserted by this rank.
        All ranks must call this method for the allgather operation to complete.
        """
        with nogil:
            deref(self._handle).insert_finished()

    def finished(self):
        """
        Check if the allgather operation has completed.

        Returns
        -------
        True if all data and finish messages have been received from all ranks,
        otherwise False.
        """
        cdef bool ret
        with nogil:
            ret = deref(self._handle).finished()
        return ret

    def wait_and_extract(self, bool ordered = True, int timeout_ms = -1):
        """
        Wait for completion and extract all gathered data.

        Blocks until the allgather operation completes and returns all
        collected data from all ranks.

        Parameters
        ----------
        ordered
            If True, returned data will be ordered first by rank and then by
            insertion order on that rank. If False, extraction is unordered.
        timeout_ms
            Maximum duration to wait in milliseconds. Negative values mean no timeout.

        Returns
        -------
        A list containing packed data from all participating ranks.

        Raises
        ------
        RuntimeError
            If the timeout is reached.
        """
        cdef vector[cpp_PackedData] _ret
        cdef milliseconds_t _timeout_ms = <milliseconds_t>timeout_ms
        cdef Ordered _ordered = <Ordered>ordered

        with nogil:
            _ret = deref(self._handle).wait_and_extract(_ordered, _timeout_ms)
        return packed_data_vector_to_list(move(_ret))

    def extract_ready(self):
        """
        Extract any available data.

        Returns
        -------
        A list containing available data (or empty if none).

        Notes
        -----
        This is a non-blocking, unordered interface. Can be used to drain
        an AllGather operation while it's still ongoing.

        Example
        -------
        >>> # Drain an AllGather
        >>> allgather = ...  # create
        >>> # ... insert data
        >>> allgather.insert_finished()  # finish inserting
        >>> results = []
        >>> while not allgather.finished():
        ...     results.extend(allgather.extract_ready())
        >>> # Extract any final chunks that may have arrived
        >>> results.extend(allgather.extract_ready())
        """
        cdef vector[cpp_PackedData] _ret
        with nogil:
            _ret = deref(self._handle).extract_ready()
        return packed_data_vector_to_list(move(_ret))
