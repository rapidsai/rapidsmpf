# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""The Shuffler interface for RapidsMPF."""

from cython.operator cimport dereference as deref
from libc.stdint cimport UINT8_MAX, uint32_t
from libcpp.memory cimport make_unique
from libcpp.unordered_map cimport unordered_map
from libcpp.utility cimport move
from libcpp.vector cimport vector
from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.pylibrmm.stream cimport Stream

from rapidsmpf.buffer.packed_data cimport PackedData, cpp_PackedData
from rapidsmpf.progress_thread cimport ProgressThread
from rapidsmpf.statistics cimport Statistics


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
    progress_thread
        The progress thread to use for tracking progress.
    op_id
        The operation ID of the shuffle. Must have a value between 0 and
        ``max_concurrent_shuffles-1``.
    total_num_partitions
        Total number of partitions in the shuffle.
    stream
        The CUDA stream used for memory operations.
    br
        The buffer resource used to allocate temporary storage and shuffle results.
    statistics
        The statistics instance to use. If None, statistics is disabled.

    Attributes
    ----------
    max_concurrent_shuffles
        Maximum number of concurrent shufflers.

    Notes
    -----
    This class is designed to handle distributed operations by partitioning data
    and redistributing it across ranks in a cluster. It is typically used in
    distributed data processing workflows involving cuDF tables.
    """
    max_concurrent_shuffles = UINT8_MAX + 1  # match the type of the `op_id` argument.

    def __init__(
        self,
        Communicator comm,
        ProgressThread progress_thread,
        uint8_t op_id,
        uint32_t total_num_partitions,
        stream,
        BufferResource br,
        Statistics statistics = None,
    ):
        if stream is None:
            raise ValueError("stream cannot be None")
        self._stream = Stream(stream)
        self._comm = comm
        self._br = br
        cdef cpp_BufferResource* br_ = br.ptr()
        cdef cuda_stream_view _stream = self._stream.view()
        if statistics is None:
            statistics = Statistics(enable=False)  # Disables statistics.
        with nogil:
            self._handle = make_unique[cpp_Shuffler](
                comm._handle,
                progress_thread._handle,
                op_id,
                total_num_partitions,
                _stream,
                br_,
                statistics._handle,
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

    def insert_chunks(self, chunks, bool grouped = False):
        """
        Insert a batch of packed (serialized) chunks into the shuffle.

        Parameters
        ----------
        chunks
            A map where keys are partition IDs (``int``) and values are packed
            data (``PackedData``).
        grouped
            If ``True``, the chunks are grouped by the destination rank of the
            partition ID.

        Notes
        -----
        This method adds the given chunks to the shuffle, associating them with their
        respective partition IDs.
        """
        # Convert python mapping to an `unordered_map`.
        cdef unordered_map[uint32_t, cpp_PackedData] _chunks
        cdef bint _grouped = grouped  # Convert Python bool to C bool
        for pid, chunk in chunks.items():
            if not (<PackedData?>chunk).c_obj:
                raise ValueError("PackedData was empty")
            _chunks[<uint32_t?>pid] = move(deref((<PackedData?>chunk).c_obj))

        with nogil:
            if _grouped:
                deref(self._handle).insert_grouped(move(_chunks))
            else:
                deref(self._handle).insert(move(_chunks))

    def insert_finished(self, uint32_t pid):
        """
        Mark a partition as finished.

        This informs the shuffler that no more chunks for the specified partition
        will be inserted.

        Parameters
        ----------
        pid
            The partition ID to mark as finished.

        Notes
        -----
        Once a partition is marked as finished, it is considered complete and no
        further chunks will be accepted for that partition.
        """
        with nogil:
            deref(self._handle).insert_finished(pid)

    def insert_finished(self, list[uint32_t] pids):
        """
        Mark a list of partitions as finished.

        This informs the shuffler that no more chunks for the specified partitions
        will be inserted.

        Parameters
        ----------
        pids
            A list of partition IDs to mark as finished.

        Notes
        -----
        Once a partition is marked as finished, it is considered complete and no
        further chunks will be accepted for that partition.
        """
        cdef vector[uint32_t] _pids
        
        # Reserve space and populate vector
        _pids.reserve(len(pids))
        for pid in pids:
            _pids.push_back(pid)

        with nogil:
            deref(self._handle).insert_finished(move(_pids))

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

        # Move the result into a python list of `PackedData`.
        cdef list ret = []
        for i in range(_ret.size()):
            ret.append(
                PackedData.from_librapidsmpf(
                    make_unique[cpp_PackedData](
                        move(_ret.at(i).metadata), move(_ret.at(i).gpu_data)
                    )
                )
            )
        return ret

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

    def wait_any(self):
        """
        Wait for any partition to finish.

        This method blocks until at least one partition is marked as finished.
        It is useful for processing partitions as they are completed.

        Returns
        -------
        The partition ID of the next finished partition.
        """
        cdef uint32_t ret
        with nogil:
            ret = deref(self._handle).wait_any()
        return ret

    def wait_on(self, uint32_t pid):
        """
        Wait for a specific partition to finish.

        This method blocks until the desired partition
        is ready for processing.

        Parameters
        ----------
        pid
            The desired partition ID.
        """
        with nogil:
            deref(self._handle).wait_on(pid)
