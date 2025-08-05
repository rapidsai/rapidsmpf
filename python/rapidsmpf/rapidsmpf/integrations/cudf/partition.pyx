# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Partitioning of cuDF tables."""

from cython.operator cimport dereference as deref
from cython.operator cimport postincrement
from libc.stdint cimport uint32_t
from libcpp cimport bool as bool_t
from libcpp.memory cimport make_unique, shared_ptr, unique_ptr
from libcpp.unordered_map cimport unordered_map
from libcpp.utility cimport move
from libcpp.vector cimport vector
from pylibcudf.libcudf.table.table cimport table as cpp_table
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.table cimport Table
from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.pylibrmm.stream cimport Stream

from rapidsmpf.buffer.packed_data cimport (PackedData, cpp_PackedData,
                                           packed_data_vector_to_list)
from rapidsmpf.buffer.resource cimport BufferResource, cpp_BufferResource
from rapidsmpf.statistics cimport cpp_Statistics


cdef extern from "<rapidsmpf/integrations/cudf/partition.hpp>" nogil:
    int cpp_HASH_MURMUR3"cudf::hash_id::HASH_MURMUR3"
    uint32_t cpp_DEFAULT_HASH_SEED"cudf::DEFAULT_HASH_SEED",

    cdef unordered_map[uint32_t, cpp_PackedData] cpp_partition_and_pack \
        "rapidsmpf::partition_and_pack"(
            const table_view& table,
            const vector[size_type] &columns_to_hash,
            int num_partitions,
            int hash_function,
            uint32_t seed,
            cuda_stream_view stream,
            cpp_BufferResource* br,
        ) except +

    cdef unordered_map[uint32_t, cpp_PackedData] cpp_split_and_pack \
        "rapidsmpf::split_and_pack"(
            const table_view& table,
            const vector[size_type] &splits,
            cuda_stream_view stream,
            cpp_BufferResource* br,
        ) except +

cpdef dict partition_and_pack(
    Table table,
    columns_to_hash,
    int num_partitions,
    stream,
    BufferResource br,
):
    """
    Partition rows from the input table into multiple packed (serialized) tables.

    Parameters
    ----------
    table
        The input table to partition.
    columns_to_hash
        Indices of the input columns to use for hashing.
    num_partitions
        The number of partitions to create.
    stream
        The CUDA stream used for memory operations.
    br
        Buffer resource for gpu data allocations.

    Returns
    -------
    A dictionary where the keys are partition IDs and the values are packed tables.

    Raises
    ------
    IndexError
        If an index in ``columns_to_hash`` is invalid.

    See Also
    --------
    rapidsmpf.integrations.cudf.partition.unpack_and_concat
    pylibcudf.partitioning.hash_partition
    pylibcudf.contiguous_split.pack
    rapidsmpf.integrations.cudf.partition.split_and_pack
    """
    if stream is None:
        raise ValueError("stream cannot be None")
    cdef cuda_stream_view _stream = Stream(stream).view()
    cdef cpp_BufferResource* _br = br.ptr()
    cdef vector[size_type] _columns_to_hash = tuple(columns_to_hash)
    cdef unordered_map[uint32_t, cpp_PackedData] _ret
    cdef table_view tbl = table.view()
    with nogil:
        _ret = cpp_partition_and_pack(
            tbl,
            _columns_to_hash,
            num_partitions,
            cpp_HASH_MURMUR3,
            cpp_DEFAULT_HASH_SEED,
            _stream,
            _br,
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
    BufferResource br,
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
    br
        Buffer resource for gpu data allocations.

    Returns
    -------
    A dictionary where the keys are partition IDs and the values are packed tables.

    Raises
    ------
    IndexError
        If the splits are out of range for ``[0, len(table)]``.

    See Also
    --------
    rapidsmpf.integrations.cudf.partition.unpack_and_concat
    pylibcudf.copying.split
    rapidsmpf.integrations.cudf.partition.partition_and_pack
    """
    if stream is None:
        raise ValueError("stream cannot be None")
    cdef cuda_stream_view _stream = Stream(stream).view()
    cdef cpp_BufferResource* _br = br.ptr()
    cdef vector[size_type] _splits = tuple(splits)
    cdef unordered_map[uint32_t, cpp_PackedData] _ret
    cdef table_view tbl = table.view()
    with nogil:
        _ret = cpp_split_and_pack(
            tbl,
            _splits,
            _stream,
            _br,
        )
    ret = {}
    cdef unordered_map[uint32_t, cpp_PackedData].iterator it = _ret.begin()
    while(it != _ret.end()):
        ret[deref(it).first] = PackedData.from_librapidsmpf(
            make_unique[cpp_PackedData](move(deref(it).second))
        )
        postincrement(it)
    return ret


cdef extern from "<rapidsmpf/integrations/cudf/partition.hpp>" nogil:
    cdef unique_ptr[cpp_table] cpp_unpack_and_concat \
        "rapidsmpf::unpack_and_concat"(
            vector[cpp_PackedData] partition,
            cuda_stream_view stream,
            cpp_BufferResource* br,
        ) except +


# Help function to convert an iterable of `PackedData` to a vector of
# `cpp_PackedData`.
cdef vector[cpp_PackedData] _partitions_py_to_cpp(partitions):
    cdef vector[cpp_PackedData] ret
    for part in partitions:
        if not (<PackedData?>part).c_obj:
            raise ValueError("PackedData was empty")
        ret.push_back(move(deref((<PackedData?>part).c_obj)))
    return move(ret)


cpdef Table unpack_and_concat(
    partitions,
    stream,
    BufferResource br,
):
    """
    Unpack (deserialize) input tables and concatenate them.

    The input partitions are consumed and are left empty on return.

    Parameters
    ----------
    partitions
        The packed input tables to unpack and concatenate.
    stream
        The CUDA stream used for memory operations.
    br
        Buffer resource for gpu data allocations.

    Returns
    -------
    The unpacked and concatenated result as a single table.

    See Also
    --------
    rapidsmpf.integrations.cudf.partition.partition_and_pack
    """
    if stream is None:
        raise ValueError("stream cannot be None")
    cdef cuda_stream_view _stream = Stream(stream).view()
    cdef cpp_BufferResource* _br = br.ptr()
    cdef vector[cpp_PackedData] _partitions = _partitions_py_to_cpp(partitions)
    cdef unique_ptr[cpp_table] _ret
    with nogil:
        _ret = cpp_unpack_and_concat(
            move(_partitions),
            _stream,
            _br,
        )
    return Table.from_libcudf(move(_ret))


cdef extern from "<rapidsmpf/integrations/cudf/partition.hpp>" nogil:
    cdef vector[cpp_PackedData] cpp_spill_partitions \
        "rapidsmpf::spill_partitions"(
            vector[cpp_PackedData] partitions,
            cuda_stream_view stream,
            cpp_BufferResource* br,
            shared_ptr[cpp_Statistics] statistics,
        ) except +


cpdef list spill_partitions(
    partitions,
    stream,
    BufferResource br,
    Statistics statistics = None,
):
    """
    Spill partitions from device memory to host memory.

    Partitions already in host memory are returned unchanged. For partitions
    in device memory, this function allocates host memory and moves the buffer
    using the provided buffer resource.

    For device-resident partitions, a host memory reservation is made before
    moving the buffer. If the reservation fails due to insufficient host memory,
    an exception is raised. Overbooking is not allowed.

    The input partitions are consumed and are left empty on return.

    Parameters
    ----------
    partitions
        The input partitions, each containing GPU or host buffers.
    stream
        The CUDA stream used for memory operations.
    br
        The buffer resource that manages memory allocation and movement.
    statistics
        The statistics instance to use. If None, statistics is disabled.

    Returns
    -------
    A list of partitions where all buffers reside in host memory.

    Raises
    ------
    MemoryError
        If host memory allocation fails. Overbooking is not allowed.
    """

    if stream is None:
        raise ValueError("stream cannot be None")
    cdef cuda_stream_view _stream = Stream(stream).view()
    cdef cpp_BufferResource* _br = br.ptr()
    cdef vector[cpp_PackedData] _partitions = _partitions_py_to_cpp(partitions)
    cdef vector[cpp_PackedData] _ret
    if statistics is None:
        statistics = Statistics(enable=False)  # Disables statistics.
    with nogil:
        _ret = cpp_spill_partitions(
            move(_partitions),
            _stream,
            _br,
            statistics._handle,
        )
    return packed_data_vector_to_list(move(_ret))


cdef extern from "<rapidsmpf/integrations/cudf/partition.hpp>" nogil:
    cdef vector[cpp_PackedData] cpp_unspill_partitions \
        "rapidsmpf::unspill_partitions"(
            vector[cpp_PackedData] partitions,
            cuda_stream_view stream,
            cpp_BufferResource* br,
            bool_t allow_overbooking,
            shared_ptr[cpp_Statistics] statistics,
        ) except +


cpdef list unspill_partitions(
    partitions,
    stream,
    BufferResource br,
    bool_t allow_overbooking,
    Statistics statistics = None,
):
    """
    Move spilled partitions (i.e., packed tables in host memory) back to device memory.

    Each partition is inspected to determine whether its buffer already resides in
    device memory. Buffers already in device memory are returned unchanged. Host-
    resident buffers are moved to device memory using the provided buffer resource
    and CUDA stream.

    If insufficient device memory is available, the buffer resource's spill manager is
    invoked to attempt to free up space. If overbooking occurs and the spill manager
    fails to reclaim enough memory, the behavior depends on the `allow_overbooking`
    flag.

    The input partitions are consumed and are left empty on return.

    Parameters
    ----------
    partitions
        The partitions to unspill, potentially containing host-resident data.
    stream
        CUDA stream used for memory operations and kernel launches.
    br
        Buffer resource responsible for memory reservation, spills, and transfers.
    allow_overbooking
        Whether to allow overbooking if not enough device memory can be reclaimed by
        spilling. If False, an error is raised if the reservation cannot be fulfilled.
    statistics
        The statistics instance to use. If None, statistics is disabled.

    Returns
    -------
    A list of partitions where each buffer resides in device memory.

    Raises
    ------
    OverflowError
        If overbooking is required but `allow_overbooking` is False and insufficient
        memory could be spilled to satisfy the reservation.
    """
    if stream is None:
        raise ValueError("stream cannot be None")
    cdef cuda_stream_view _stream = Stream(stream).view()
    cdef cpp_BufferResource* _br = br.ptr()
    cdef vector[cpp_PackedData] _partitions = _partitions_py_to_cpp(partitions)
    cdef vector[cpp_PackedData] _ret
    if statistics is None:
        statistics = Statistics(enable=False)  # Disables statistics.
    with nogil:
        _ret = cpp_unspill_partitions(
            move(_partitions),
            _stream,
            _br,
            allow_overbooking,
            statistics._handle,
        )
    return packed_data_vector_to_list(move(_ret))
