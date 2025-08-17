# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference as deref
from libc.stddef cimport size_t
from libc.stdint cimport uint64_t
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.table.table_view cimport table_view as cpp_table_view
from pylibcudf.table cimport Table
from rmm.librmm.cuda_stream_view cimport cuda_stream_view


cdef class TableChunk:
    def __init__(
        self,
        uint64_t sequence_number,
        Table table,
        stream,
    ):
        raise ValueError("use the `from_*` factory functions")

    @staticmethod
    cdef TableChunk from_handle(
        unique_ptr[cpp_TableChunk] handle, Stream stream, object owner
    ):
        cdef TableChunk ret = TableChunk.__new__(TableChunk)
        ret._handle = move(handle)
        ret._stream = stream
        ret._owner = owner
        return ret

    @staticmethod
    def from_pylibcudf_table(uint64_t sequence_number, Table table, stream):
        if stream is None:
            raise ValueError("stream cannot be None")
        cdef Stream _stream = Stream(stream)
        cdef cuda_stream_view _stream_view = _stream.view()

        cdef size_t device_alloc_size = 0
        for col in table.columns():
            device_alloc_size += (<Column?>col).device_buffer_size()

        cdef cpp_table_view view = table.view()
        cdef unique_ptr[cpp_TableChunk] ret
        with nogil:
            ret = make_unique[cpp_TableChunk](
                sequence_number,
                view,
                device_alloc_size,
                _stream_view
            )
        return TableChunk.from_handle(move(ret), _stream, table)

    def __dealloc__(self):
        with nogil:
            self._handle.reset()

    def sequence_number(self):
        return deref(self._handle).sequence_number()

    def stream(self):
        if self._stream is None:
            self._stream = Stream._from_cudaStream_t(
                deref(self._handle).stream().value()
            )
        return self._stream

    def data_alloc_size(self, MemoryType mem_type):
        return deref(self._handle).data_alloc_size(mem_type)

    def is_available(self):
        return deref(self._handle).is_available()

    def table_view(self):
        cdef cpp_table_view ret
        with nogil:
            ret = deref(self._handle).table_view()
        return Table.from_table_view_of_arbitrary(ret, owner=self)
