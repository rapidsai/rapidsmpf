# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport shared_ptr
from rmm.pylibrmm.stream cimport Stream

from rapidsmpf.buffer.resource cimport BufferResource
from rapidsmpf.communicator.communicator cimport Communicator
from rapidsmpf.config cimport Options
from rapidsmpf.statistics cimport Statistics


cdef extern from "<rapidsmpf/streaming/core/context.hpp>" nogil:
    cdef cppclass cpp_Context "rapidsmpf::streaming::Context":
        pass


cdef class Context:
    cdef shared_ptr[cpp_Context] _handle
    cdef Communicator _comm
    cdef BufferResource _br
    cdef Options _options
    cdef Statistics _statistics
