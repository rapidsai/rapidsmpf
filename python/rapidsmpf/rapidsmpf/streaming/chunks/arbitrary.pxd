# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr

from rapidsmpf.owning_wrapper cimport cpp_OwningWrapper


cdef class ArbitraryChunk:
    cdef unique_ptr[cpp_OwningWrapper] _handle

    @staticmethod
    cdef ArbitraryChunk from_handle(unique_ptr[cpp_OwningWrapper] handle)
    cdef const cpp_OwningWrapper* handle_ptr(self)
    cdef unique_ptr[cpp_OwningWrapper] release_handle(self)
