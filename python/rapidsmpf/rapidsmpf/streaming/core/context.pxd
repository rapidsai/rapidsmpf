# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport shared_ptr

from rapidsmpf.communicator.communicator cimport Communicator
from rapidsmpf.config cimport Options
from rapidsmpf.memory.buffer cimport MemoryType
from rapidsmpf.memory.buffer_resource cimport BufferResource
from rapidsmpf.statistics cimport Statistics
from rapidsmpf.streaming.core.channel cimport cpp_Channel
from rapidsmpf.streaming.core.memory_reserve_or_wait cimport \
    cpp_MemoryReserveOrWait
from rapidsmpf.streaming.core.spillable_messages cimport (
    SpillableMessages, cpp_SpillableMessages)


cdef extern from "<rapidsmpf/streaming/core/context.hpp>" nogil:
    cdef cppclass cpp_Context "rapidsmpf::streaming::Context":
        shared_ptr[cpp_Channel] create_channel() except +
        shared_ptr[cpp_SpillableMessages] spillable_messages() noexcept
        void shutdown() noexcept
        shared_ptr[cpp_MemoryReserveOrWait] memory(MemoryType mem_type) noexcept


cdef class Context:
    cdef shared_ptr[cpp_Context] _handle
    cdef Communicator _comm
    cdef BufferResource _br
    cdef Options _options
    cdef Statistics _statistics
    cdef SpillableMessages _spillable_messages
    cdef dict _memory
