# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Partitioning of cuDF tables."""

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
            device_memory_resource *mr,
        ) except +

    cdef unordered_map[uint32_t, cpp_PackedData] cpp_split_and_pack \
        "rapidsmpf::split_and_pack"(
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
    num_partitions
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
    rapidsmpf.integrations.cudf.partition.unpack_and_concat
    pylibcudf.partitioning.hash_partition
    pylibcudf.contiguous_split.pack
    rapidsmpf.integrations.cudf.partition.split_and_pack
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
    IndexError
        If the splits are out of range for ``[0, len(table)]``.

    See Also
    --------
    rapidsmpf.integrations.cudf.partition.unpack_and_concat
    pylibcudf.copying.split
    rapidsmpf.integrations.cudf.partition_and_pack
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


cdef extern from "<rapidsmpf/integrations/cudf/partition.hpp>" nogil:
    cdef unique_ptr[cpp_table] cpp_unpack_and_concat \
        "rapidsmpf::unpack_and_concat"(
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
    rapidsmpf.integrations.cudf.partition_and_pack
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
