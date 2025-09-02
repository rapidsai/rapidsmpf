# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector


cdef extern from "<rapidsmpf/streaming/core/node.hpp>" nogil:
    cdef cppclass cpp_Node "rapidsmpf::streaming::Node":
        pass

    cdef void cpp_run_streaming_pipeline \
        "rapidsmpf::streaming::run_streaming_pipeline"(vector[cpp_Node]) except +


cdef class CppNode:
    cdef unique_ptr[cpp_Node] _handle
    cdef object _owner

    @staticmethod
    cdef CppNode from_handle(unique_ptr[cpp_Node] handle, object owner)

    cdef unique_ptr[cpp_Node] release_handle(self)
