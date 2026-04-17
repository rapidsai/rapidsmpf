# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr

from rapidsmpf.memory.buffer_resource cimport BufferResource
from rapidsmpf.memory.packed_data cimport cpp_PackedData


cdef class PackedDataChunk:
    cdef unique_ptr[cpp_PackedData] _handle
    # Keep the BufferResource alive as long as this object is so that when this
    # object is deallocated the associated stream and memory resource are still alive.
    cdef BufferResource _br

    @staticmethod
    cdef PackedDataChunk from_handle(
        unique_ptr[cpp_PackedData] handle, BufferResource br
    )
    cdef const cpp_PackedData* handle_ptr(self)
    cdef unique_ptr[cpp_PackedData] release_handle(self)
