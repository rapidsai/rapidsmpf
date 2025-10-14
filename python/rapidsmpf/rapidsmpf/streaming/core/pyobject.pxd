# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cpython.object cimport PyObject
from libc.stdint cimport uint64_t
from libcpp.memory cimport unique_ptr

from rapidsmpf.streaming.core.channel cimport cpp_Message


cdef extern from *:
    cdef cppclass cpp_PyObjectPayload "rapidsmpf::streaming::PyObjectPayload":
        uint64_t sequence_number
        PyObject* py_obj
        cpp_PyObjectPayload(uint64_t seq, PyObject* obj) except +

    unique_ptr[cpp_PyObjectPayload] \
        cpp_release_pyobject_from_message(cpp_Message) except +


cdef class PyObjectPayload:
    cdef unique_ptr[cpp_PyObjectPayload] _handle

    @staticmethod
    cdef PyObjectPayload from_handle(unique_ptr[cpp_PyObjectPayload] handle)
    cdef const cpp_PyObjectPayload* handle_ptr(self)
    cdef unique_ptr[cpp_PyObjectPayload] release_handle(self)
