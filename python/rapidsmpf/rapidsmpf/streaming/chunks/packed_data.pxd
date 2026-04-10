# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr

from rapidsmpf.memory.buffer_resource cimport BufferResource
from rapidsmpf.memory.packed_data cimport cpp_PackedData


cdef class PackedDataChunk:
    cdef unique_ptr[cpp_PackedData] _handle
    # Prevent the BufferResource (and its stream) from being garbage collected.
    cdef BufferResource _br

    @staticmethod
    cdef PackedDataChunk from_handle(
        unique_ptr[cpp_PackedData] handle, BufferResource br=*
    )
    cdef const cpp_PackedData* handle_ptr(self)
    cdef unique_ptr[cpp_PackedData] release_handle(self)
