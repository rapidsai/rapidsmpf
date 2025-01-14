# Copyright (c) 2025, NVIDIA CORPORATION.

from cython.operator cimport dereference as deref
from cython.operator cimport postincrement
from libc.stdint cimport uint32_t
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.unordered_map cimport unordered_map
from libcpp.utility cimport move
from libcpp.vector cimport vector
from pylibcudf.contiguous_split cimport PackedColumns
from pylibcudf.libcudf.contiguous_split cimport packed_columns
from pylibcudf.libcudf.table.table cimport table as cpp_table
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.table cimport Table
from rmm._cuda.stream cimport Stream


cdef class Shuffler:
    def __init__(
        self,
        Communicator comm,
        uint32_t total_num_partitions,
        stream,
        BufferResource br,
    ):
        self._stream = Stream(stream)
        self._comm = comm
        self._br = br
        self._handle = make_unique[cpp_Shuffler](
            comm._handle, total_num_partitions, self._stream.view(), br.ptr()
        )

    @property
    def comm(self):
        return self._comm


cdef extern from "<rapidsmp/shuffler/partition.hpp>" nogil:
    cdef unordered_map[uint32_t, packed_columns] cpp_partition_and_pack \
        "rapidsmp::shuffler::partition_and_pack"(
            const table_view& table,
            const vector[size_type] &columns_to_hash,
            int num_partitions,
        ) except +


cpdef dict partition_and_pack(Table table, columns_to_hash, int num_partitions):
    cdef vector[size_type] _columns_to_hash = tuple(columns_to_hash)
    cdef unordered_map[uint32_t, packed_columns] _ret
    cdef table_view tbl = table.view()
    with nogil:
        _ret = cpp_partition_and_pack(tbl, _columns_to_hash, num_partitions)
    ret = {}
    cdef unordered_map[uint32_t, packed_columns].iterator it = _ret.begin()
    while(it != _ret.end()):
        ret[deref(it).first] = PackedColumns.from_libcudf(
            make_unique[packed_columns](move(deref(it).second))
        )
        postincrement(it)
    return ret


cdef extern from "<rapidsmp/shuffler/partition.hpp>" nogil:
    cdef unique_ptr[cpp_table] cpp_unpack_and_concat \
        "rapidsmp::shuffler::unpack_and_concat"(
            vector[packed_columns] partition
        ) except +


cpdef Table unpack_and_concat(partitions):
    cdef vector[packed_columns] _partitions
    for part in partitions:
        _partitions.push_back(move(deref((<PackedColumns?>part).c_obj)))
    cdef unique_ptr[cpp_table] _ret
    with nogil:
        _ret = cpp_unpack_and_concat(move(_partitions))
    return Table.from_libcudf(move(_ret))
