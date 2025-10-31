# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr


cdef extern from "<rapidsmpf/streaming/cudf/owning_wrapper.hpp>" nogil:
    cdef cppclass cpp_OwningWrapper "rapidsmpf::streaming::OwningWrapper":
        cpp_OwningWrapper(void *, void(*)(void*)) noexcept
        void* release() noexcept


cdef class ArbitraryChunk:
    cdef unique_ptr[cpp_OwningWrapper] _handle

    @staticmethod
    cdef ArbitraryChunk from_handle(unique_ptr[cpp_OwningWrapper] handle)
    cdef const cpp_OwningWrapper* handle_ptr(self)
    cdef unique_ptr[cpp_OwningWrapper] release_handle(self)
