# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""The Shuffler interface for RapidsMPF."""

from cython.operator cimport dereference as deref
from cython.operator cimport postincrement
from libc.stdint cimport uint32_t
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.unordered_map cimport unordered_map
from libcpp.utility cimport move
from libcpp.vector cimport vector
from pylibcudf.libcudf.table.table cimport table as cpp_table
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.table cimport Table
from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.librmm.memory_resource cimport device_memory_resource
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

from rapidsmpf.buffer.packed_data cimport PackedData, cpp_PackedData
from rapidsmpf.progress_thread cimport ProgressThread
from rapidsmpf.statistics cimport Statistics


cdef extern from "<rapidsmpf/shuffler/partition.hpp>" nogil:
    int cpp_HASH_MURMUR3"cudf::hash_id::HASH_MURMUR3"
    uint32_t cpp_DEFAULT_HASH_SEED"cudf::DEFAULT_HASH_SEED",

    cdef unordered_map[uint32_t, cpp_PackedData] cpp_partition_and_pack \
        "rapidsmpf::shuffler::partition_and_pack"(
            const table_view& table,
            const vector[size_type] &columns_to_hash,
            int num_partitions,
            int hash_function,
            uint32_t seed,
            cuda_stream_view stream,
            device_memory_resource *mr,
        ) except +

    cdef unordered_map[uint32_t, cpp_PackedData] cpp_split_and_pack \
        "rapidsmpf::shuffler::split_and_pack"(
            const table_view& table,
            const vector[size_type] &splits,
            cuda_stream_view stream,
            device_memory_resource *mr,
        ) except +


cpdef dict partition_and_pack(
    Table table,
    columns_to_hash,
    int num_partitions,
    stream,
    DeviceMemoryResource device_mr,
):
    """
    Partition rows from the input table into multiple packed (serialized) tables.

    Parameters
    ----------
    table
        The input table to partition.
    columns_to_hash
        Indices of the input columns to use for hashing.
    num_partitions : int
        The number of partitions to create.
    stream
        The CUDA stream used for memory operations.
    device_mr
        Reference to the RMM device memory resource used for device allocations.

    Returns
    -------
    A dictionary where the keys are partition IDs and the values are packed tables.

    Raises
    ------
    IndexError
        If an index in ``columns_to_hash`` is invalid.

    See Also
    --------
    rapidsmpf.shuffler.unpack_and_concat
    pylibcudf.partitioning.hash_partition
    pylibcudf.contiguous_split.pack
    pylibcudf.partitioning.split_and_pack
    """
    cdef vector[size_type] _columns_to_hash = tuple(columns_to_hash)
    cdef unordered_map[uint32_t, cpp_PackedData] _ret
    cdef table_view tbl = table.view()
    if stream is None:
        raise ValueError("stream cannot be None")
    cdef cuda_stream_view _stream = Stream(stream).view()
    with nogil:
        _ret = cpp_partition_and_pack(
            tbl,
            _columns_to_hash,
            num_partitions,
            cpp_HASH_MURMUR3,
            cpp_DEFAULT_HASH_SEED,
            _stream,
            device_mr.get_mr()
        )
    ret = {}
    cdef unordered_map[uint32_t, cpp_PackedData].iterator it = _ret.begin()
    while(it != _ret.end()):
        ret[deref(it).first] = PackedData.from_librapidsmpf(
            make_unique[cpp_PackedData](move(deref(it).second))
        )
        postincrement(it)
    return ret


cpdef dict split_and_pack(
    Table table,
    splits,
    stream,
    DeviceMemoryResource device_mr,
):
    """
    Splits rows from the input table into multiple packed (serialized) tables.

    Parameters
    ----------
    table
        The input table to split and pack.  The table cannot be empty (the
        split points would not be valid).
    splits
        The split points, equivalent to cudf::split(), i.e. one less than
        the number of result partitions.
    stream
        The CUDA stream used for memory operations.
    device_mr
        Reference to the RMM device memory resource used for device allocations.

    Returns
    -------
    A dictionary where the keys are partition IDs and the values are packed tables.

    Raises
    ------
    ValueError
        If the input table is empty.

    See Also
    --------
    rapidsmp.shuffler.unpack_and_concat
    pylibcudf.copy.split
    pylibcudf.partitioning.partition_and_pack
    """
    cdef vector[size_type] _splits = tuple(splits)
    cdef unordered_map[uint32_t, cpp_PackedData] _ret
    cdef table_view tbl = table.view()
    if stream is None:
        raise ValueError("stream cannot be None")
    cdef cuda_stream_view _stream = Stream(stream).view()

    with nogil:
        _ret = cpp_split_and_pack(
            tbl,
            _splits,
            _stream,
            device_mr.get_mr()
        )
    ret = {}
    cdef unordered_map[uint32_t, cpp_PackedData].iterator it = _ret.begin()
    while(it != _ret.end()):
        ret[deref(it).first] = PackedData.from_librapidsmpf(
            make_unique[cpp_PackedData](move(deref(it).second))
        )
        postincrement(it)
    return ret


cdef extern from "<rapidsmpf/shuffler/partition.hpp>" nogil:
    cdef unique_ptr[cpp_table] cpp_unpack_and_concat \
        "rapidsmpf::shuffler::unpack_and_concat"(
            vector[cpp_PackedData] partition,
            cuda_stream_view stream,
            device_memory_resource *mr,
        ) except +


cpdef Table unpack_and_concat(
    partitions,
    stream,
    DeviceMemoryResource device_mr,
):
    """
    Unpack (deserialize) input tables and concatenate them.

    Parameters
    ----------
    partitions
        The packed input tables to unpack and concatenate.
    stream
        The CUDA stream used for memory operations.
    device_mr
        Reference to the RMM device memory resource used for device allocations.

    Returns
    -------
    The unpacked and concatenated result as a single table.

    See Also
    --------
    rapidsmpf.shuffler.partition_and_pack
    """
    cdef vector[cpp_PackedData] _partitions
    for part in partitions:
        if not (<PackedData?>part).c_obj:
            raise ValueError("PackedData was empty")
        _partitions.push_back(move(deref((<PackedData?>part).c_obj)))
    if stream is None:
        raise ValueError("stream cannot be None")
    cdef cuda_stream_view _stream = Stream(stream).view()
    cdef unique_ptr[cpp_table] _ret
    with nogil:
        _ret = cpp_unpack_and_concat(
            move(_partitions),
            _stream,
            device_mr.get_mr()
        )
    return Table.from_libcudf(move(_ret))


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
        The operation ID of the shuffle.
    total_num_partitions
        Total number of partitions in the shuffle.
    stream
        The CUDA stream used for memory operations.
    br
        The buffer resource used to allocate temporary storage and shuffle results.
    statistics
        The statistics instance to use. If None, statistics is disabled.

    Notes
    -----
    This class is designed to handle distributed operations by partitioning data
    and redistributing it across ranks in a cluster. It is typically used in
    distributed data processing workflows involving cuDF tables.
    """
    def __init__(
        self,
        Communicator comm,
        ProgressThread progress_thread,
        uint16_t op_id,
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
        if statistics is None:
            statistics = Statistics(enable=False)  # Disables statistics.
        with nogil:
            self._handle = make_unique[cpp_Shuffler](
                comm._handle,
                progress_thread._handle,
                op_id,
                total_num_partitions,
                self._stream.view(),
                br_,
                statistics._handle,
            )

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
        # Convert python mapping to an `unordered_map`.
        cdef unordered_map[uint32_t, cpp_PackedData] _chunks
        for pid, chunk in chunks.items():
            if not (<PackedData?>chunk).c_obj:
                raise ValueError("PackedData was empty")
            _chunks[<uint32_t?>pid] = move(deref((<PackedData?>chunk).c_obj))

        with nogil:
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
