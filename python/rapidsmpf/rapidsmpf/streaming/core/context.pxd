# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport shared_ptr

from rapidsmpf.buffer.resource cimport BufferResource
from rapidsmpf.communicator.communicator cimport Communicator
from rapidsmpf.config cimport Options
from rapidsmpf.statistics cimport Statistics
from rapidsmpf.streaming.core.channel cimport cpp_Channel


cdef extern from "<rapidsmpf/streaming/core/context.hpp>" nogil:
    cdef cppclass cpp_Context "rapidsmpf::streaming::Context":
        shared_ptr[cpp_Channel] create_channel() except +


cdef class Context:
    cdef shared_ptr[cpp_Context] _handle
    cdef Communicator _comm
    cdef BufferResource _br
    cdef Options _options
    cdef Statistics _statistics
