# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""The Shuffler interface for RapidsMPF."""

from cython.operator cimport dereference as deref
from libc.stdint cimport uint32_t
from libcpp.memory cimport make_unique
from libcpp.span cimport span
from libcpp.unordered_map cimport unordered_map
from libcpp.utility cimport move
from libcpp.vector cimport vector

from rapidsmpf.memory.packed_data cimport (PackedData, cpp_PackedData,
                                           packed_data_vector_to_list)


cdef class Shuffler:
    """
    Shuffle service for partitioned data.

    The `rapidsmpf.shuffler.Shuffler` class provides an interface for
    performing a shuffle operation on partitioned data. It uses a
    distribution scheme to distribute and collect data chunks across
    different ranks.

    Parameters
    ----------
    comm
        The communicator to use for data exchange between ranks.
    op_id
        The operation ID of the shuffle. Must have a value between 0 and
        ``max_concurrent_shuffles-1``.
    total_num_partitions
        Total number of partitions in the shuffle.
    br
        The buffer resource used to allocate temporary storage and shuffle results.
    partition_assignment
        How to assign partition IDs to ranks: :attr:`~.PartitionAssignment.ROUND_ROBIN`
        (default) for load balance (e.g. hash shuffle), or
        :attr:`~.PartitionAssignment.CONTIGUOUS` so each rank gets a contiguous range
        of partition IDs (e.g. for sort so concatenation order matches global order).
        A custom callable may be supported in the future.

    Attributes
    ----------
    max_concurrent_shuffles
        Maximum number of concurrent shufflers.

    Notes
    -----
    This class is designed to handle distributed operations by partitioning data
    and redistributing it across ranks in a cluster. It is typically used in
    distributed data processing workflows involving cuDF tables.

    The caller promises that inserted buffers are stream-ordered with respect to
    their own stream, and extracted buffers are likewise guaranteed to be stream-
    ordered with respect to their own stream.
    """
    max_concurrent_shuffles = 1 << 20  # See docs in communicator.hpp

    def __init__(
        self,
        Communicator comm not None,
        int32_t op_id,
        uint32_t total_num_partitions,
        BufferResource br not None,
        PartitionAssignment partition_assignment = PartitionAssignment.ROUND_ROBIN,
    ):
        self._br = br
        self._comm = comm
        cdef cpp_BufferResource* br_ = br.ptr()
        with nogil:
            self._handle = make_unique[cpp_Shuffler](
                comm._handle,
                op_id,
                total_num_partitions,
                br_,
                cpp_Shuffler.round_robin
                if partition_assignment == PartitionAssignment.ROUND_ROBIN
                else cpp_Shuffler.contiguous,
            )

    def __dealloc__(self):
        with nogil:
            self._handle.reset()

    def shutdown(self):
        """
        Shutdown the shuffle, blocking until all inflight communication is completed.

        Raises
        ------
        RuntimeError
            If the shuffler is already inactive.

        Notes
        -----
        This method ensures that all pending shuffle operations and communications
        are completed before shutting down. It blocks until no inflight operations
        remain.
        """
        with nogil:
            deref(self._handle).shutdown()

    def __str__(self):
        return deref(self._handle).str().decode('UTF-8')

    @property
    def comm(self):
        """
        Get the communicator used by the shuffler.

        Returns
        -------
        The communicator.
        """
        return self._comm

    def insert_chunks(self, chunks):
        """
        Insert a batch of packed (serialized) chunks into the shuffle.

        Parameters
        ----------
        chunks
            A map where keys are partition IDs (``int``) and values are packed
            data (``PackedData``).

        Notes
        -----
        This method adds the given chunks to the shuffle, associating them with their
        respective partition IDs.
        """
        cdef unordered_map[uint32_t, cpp_PackedData] _chunks

        _chunks.reserve(len(chunks))
        for pid, chunk in chunks.items():
            if not (<PackedData?>chunk).c_obj:
                raise ValueError("PackedData was empty")
            cpp_insert_chunk_into_partition_map(
                _chunks, <uint32_t?>pid, move((<PackedData?>chunk).c_obj)
            )

        with nogil:
            deref(self._handle).insert(move(_chunks))

    def insert_finished(self):
        """
        Signal that no more data will be inserted into the shuffle.

        This informs the shuffler that this rank has finished inserting
        data. Must be called exactly once.
        """
        with nogil:
            deref(self._handle).insert_finished()

    def extract(self, uint32_t pid):
        """
        Extract all chunks of the specified partition.

        Parameters
        ----------
        pid
            The partition ID to extract chunks for.

        Returns
        -------
        A list of packed data belonging to the specified partition.
        """
        cdef vector[cpp_PackedData] _ret
        with nogil:
            _ret = deref(self._handle).extract(pid)
        return packed_data_vector_to_list(move(_ret), self._br)

    def finished(self):
        """
        Check if all partitions are finished.

        This method verifies if all partitions have been completed, meaning all
        chunks have been inserted and no further data is expected from neither
        the local nor any remote nodes.

        Returns
        -------
        True if all partitions are finished, otherwise False.
        """
        cdef bool ret
        with nogil:
            ret = deref(self._handle).finished()
        return ret

    def wait(self):
        """
        Wait for all partitions to finish (blocking).

        This method blocks until all partitions are finished and ready
        to be extracted.
        """
        with nogil:
            deref(self._handle).wait()

    def local_partitions(self):
        """
        Return the partition IDs owned by this rank.

        Returns
        -------
        Partition IDs owned by this shuffler.
        """
        cdef span[const uint32_t] _ret
        cdef list partitions = []
        with nogil:
            _ret = deref(self._handle).local_partitions()
        for pid in _ret:
            partitions.append(pid)
        return partitions
