# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libc.stddef cimport size_t
from libc.stdint cimport uint64_t
from libcpp cimport bool as bool_t
from libcpp.memory cimport unique_ptr
from pylibcudf.libcudf.table.table_view cimport table_view as cpp_table_view
from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.pylibrmm.stream cimport Stream

from rapidsmpf.buffer.buffer cimport MemoryType


cdef extern from "<rapidsmpf/streaming/cudf/table_chunk.hpp>" nogil:
    cdef cppclass cpp_TableChunk "rapidsmpf::streaming::TableChunk":
        cuda_stream_view stream() noexcept
        size_t data_alloc_size(MemoryType mem_type) except +
        bool_t is_available() noexcept
        size_t make_available_cost() noexcept
        cpp_table_view table_view() except +
        bool_t is_spillable() noexcept

cdef class TableChunk:
    cdef unique_ptr[cpp_TableChunk] _handle

    @staticmethod
    cdef TableChunk from_handle(unique_ptr[cpp_TableChunk] handle)
    cdef const cpp_TableChunk* handle_ptr(self)
    cdef unique_ptr[cpp_TableChunk] release_handle(self)
