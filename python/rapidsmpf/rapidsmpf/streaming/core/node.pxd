# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector

from rapidsmpf._detail.exception_handling cimport ex_handler


cdef extern from "<rapidsmpf/streaming/core/node.hpp>" nogil:
    cdef cppclass cpp_Actor "rapidsmpf::streaming::Actor":
        pass

    cdef void cpp_run_actor_graph \
        "rapidsmpf::streaming::run_actor_graph"(vector[cpp_Actor]) \
        except +ex_handler


cdef class CppActor:
    cdef unique_ptr[cpp_Actor] _handle
    cdef object _owner

    @staticmethod
    cdef CppActor from_handle(unique_ptr[cpp_Actor] handle, object owner)

    cdef unique_ptr[cpp_Actor] release_handle(self)
