# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Sparse alltoall interface for RapidsMPF."""

from cython.operator cimport dereference as deref
from libc.stdint cimport int32_t
from libcpp.memory cimport make_unique
from libcpp.utility cimport move
from libcpp.vector cimport vector

from rapidsmpf.coll.sparse_alltoall cimport cpp_SparseAlltoall, milliseconds_t
from rapidsmpf.communicator.communicator cimport Communicator, Rank
from rapidsmpf.memory.buffer_resource cimport (BufferResource,
                                               cpp_BufferResource)
from rapidsmpf.memory.packed_data cimport (PackedData, cpp_PackedData,
                                           packed_data_vector_to_list)


cdef class SparseAlltoall:
    """
    Sparse all-to-all collective over explicit source and destination peer sets.

    Each rank may send zero or more messages to ranks listed in `dsts` and
    receives zero or more messages from ranks listed in `srcs`. Sender
    order is defined by the local order of calls to `insert(dst, ...)` for
    each destination rank.

    This object is logically collective over the communicator and
    identified by `op_id`. Local extraction is only valid after `wait()`
    has completed.

    Parameters
    ----------
    comm
        The communicator for communication between ranks.
    op_id
        Unique operation identifier for this collective. Must have a value
        between 0 and 2^20 - 1.
    br
        Buffer resource for memory allocation.
    srcs
        Sources to receive from
    dsts
        Destinations to send to

    Notes
    -----
    The caller promises that inserted buffers are stream-ordered with respect to
    their own stream, and extracted buffers are likewise guaranteed to be stream-
    ordered with respect to their own stream.
    """

    def __init__(
        self,
        Communicator comm not None,
        int32_t op_id,
        BufferResource br not None,
        srcs,
        dsts
    ):
        self._br = br
        self._comm = comm
        cdef cpp_BufferResource* br_ = br.ptr()
        cdef vector[Rank] c_srcs = list(srcs)
        cdef vector[Rank] c_dsts = list(dsts)
        with nogil:
            self._handle = make_unique[cpp_SparseAlltoall](
                comm._handle,
                op_id,
                br_,
                move(c_srcs),
                move(c_dsts),
            )

    def __dealloc__(self):
        with nogil:
            self._handle.reset()

    @property
    def comm(self):
        """
        Get the communicator used by the alltoall.

        Returns
        -------
        The communicator.
        """
        return self._comm

    def insert(self, Rank dst, PackedData packed_data):
        """
        Insert packed data into the collective (non-blocking).

        Parameters
        ----------
        dst
            The destination rank for the data.
        packed_data
            The data to contribute.
        """
        if not packed_data.c_obj:
            raise ValueError("PackedData was empty")

        with nogil:
            deref(self._handle).insert(dst, move(deref(packed_data.c_obj)))

    def insert_finished(self):
        """
        Mark that this rank has finished contributing data.

        Notes
        -----
        This method signals that no more data will be inserted by this rank.
        All ranks must call this method for the collective to complete.
        """
        with nogil:
            deref(self._handle).insert_finished()

    def wait(self, int timeout_ms = -1):
        """
        Wait for completion.

        Parameters
        ----------
        timeout_ms
            Maximum duration to wait in milliseconds. Negative values mean no timeout.

        Raises
        ------
        RuntimeError
            If the timeout is reached.
        """
        cdef milliseconds_t _timeout_ms = <milliseconds_t>timeout_ms

        with nogil:
            deref(self._handle).wait(_timeout_ms)

    def extract(self, Rank src):
        """
        Extract data from given source.

        Parameters
        ----------
        src
            Source to extract data from.

        Returns
        -------
        list[PackedData]
            List of the messages received from src.

        Notes
        -----
        `wait` must have successfully returned before you call this function.
        """
        cdef vector[cpp_PackedData] c_ret
        with nogil:
            c_ret = deref(self._handle).extract(src)
        return packed_data_vector_to_list(move(c_ret), self._br)
