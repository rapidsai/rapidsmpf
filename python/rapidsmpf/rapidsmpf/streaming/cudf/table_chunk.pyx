# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference as deref
from libc.stddef cimport size_t
from libc.stdint cimport uint64_t
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector
from pylibcudf.column cimport Column
from pylibcudf.libcudf.table.table_view cimport table_view as cpp_table_view
from pylibcudf.table cimport Table
from rmm.librmm.cuda_stream_view cimport cuda_stream_view

from rapidsmpf.streaming.core.context cimport Context
from rapidsmpf.streaming.core.leaf_node cimport (cpp_pull_chunks_from_channel,
                                                 cpp_push_chunks_to_channel)
from rapidsmpf.streaming.core.node cimport Node, cpp_Node


cdef class TableChunk:
    def __init__(self):
        raise ValueError("use the `from_*` factory functions")

    @staticmethod
    cdef TableChunk from_handle(
        unique_ptr[cpp_TableChunk] handle, Stream stream, object owner
    ):
        if stream is None:
            stream = Stream._from_cudaStream_t(
                deref(handle).stream().value()
            )
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

    cdef const cpp_TableChunk* handle_ptr(self):
        if not self._handle:
            raise ValueError("TableChunk is uninitialized, has it been consumed?")
        return self._handle.get()

    cdef unique_ptr[cpp_TableChunk] handle_release(self):
        if not self._handle:
            raise ValueError("TableChunk is uninitialized, has it been consumed?")
        return move(self._handle)

    def sequence_number(self):
        return deref(self.handle_ptr()).sequence_number()

    def stream(self):
        return self._stream

    def data_alloc_size(self, MemoryType mem_type):
        return deref(self.handle_ptr()).data_alloc_size(mem_type)

    def is_available(self):
        return deref(self.handle_ptr()).is_available()

    def table_view(self):
        cdef const cpp_TableChunk* handle = self.handle_ptr()
        cdef cpp_table_view ret
        with nogil:
            ret = deref(handle).table_view()
        return Table.from_table_view_of_arbitrary(ret, owner=self)


cdef class TableChunkChannel:
    @staticmethod
    cdef TableChunkChannel from_handle(cpp_SharedChannel[cpp_TableChunk] handle):
        cdef TableChunkChannel ret = TableChunkChannel.__new__(TableChunkChannel)
        ret._handle = move(handle)
        return ret

    def __dealloc__(self):
        with nogil:
            self._handle.reset()


def push_table_chunks_to_channel(Context ctx, TableChunkChannel ch_out, list chunks):
    # Warning chunks is consumed
    cdef vector[unique_ptr[cpp_TableChunk]] _chunks
    cdef TableChunk _chunk
    owner = []
    for chunk in chunks:
        _chunk = <TableChunk?>chunk
        owner.append(_chunk._owner)
        owner.append(_chunk._stream)
        _chunks.emplace_back(move(_chunk.handle_release()))
    chunks.clear()

    cdef cpp_Node _ret
    with nogil:
        _ret = cpp_push_chunks_to_channel[cpp_TableChunk](
            ctx._handle, ch_out._handle, move(_chunks)
        )
    return Node.from_handle(move(_ret), owner)


cdef class DeferredOutputChunks:
    def result(self):
        cdef list ret = []
        for i in range(self._chunks.size()):
            # TODO: need some locking.
            ret.append(
                TableChunk.from_handle(
                    handle=move(self._chunks[i]),
                    stream=None,
                    owner=None,
                )
            )
        return ret


def pull_chunks_from_channel(
    Context ctx, TableChunkChannel ch_in, DeferredOutputChunks chunks
):
    cdef cpp_Node _ret_node
    with nogil:
        _ret_node = cpp_pull_chunks_from_channel[cpp_TableChunk](
            ctx._handle, ch_in._handle, chunks._chunks
        )
    return Node.from_handle(move(_ret_node), owner=None)
