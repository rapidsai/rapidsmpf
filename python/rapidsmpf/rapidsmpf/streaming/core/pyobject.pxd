# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cpython.object cimport PyObject
from libc.stdint cimport uint64_t
from libcpp.memory cimport unique_ptr

from rapidsmpf.streaming.core.channel cimport cpp_Message


cdef extern from *:
    cdef cppclass cpp_OwningWrapper "rapidsmpf::streaming::OwningWrapper":
        cpp_OwningWrapper() except +
        cpp_OwningWrapper(void*, void (*)(void*)) except +
        void* release() except +

    cdef cppclass cpp_TypeErasedChunk "rapidsmpf::streaming::TypeErasedChunk":
        uint64_t sequence_
        cpp_OwningWrapper obj_  # Public member for accessing wrapped object
        cpp_TypeErasedChunk() except +
        cpp_TypeErasedChunk(cpp_OwningWrapper, uint64_t) except +
        void* release() except +

    unique_ptr[cpp_TypeErasedChunk] cpp_release_from_message(cpp_Message) except +


cdef class PyObjectPayload:
    cdef unique_ptr[cpp_TypeErasedChunk] _handle

    @staticmethod
    cdef PyObjectPayload from_handle(unique_ptr[cpp_TypeErasedChunk] handle)
    cdef const cpp_TypeErasedChunk* handle_ptr(self)
    cdef unique_ptr[cpp_TypeErasedChunk] release_handle(self)
