# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int32_t
from libcpp.memory cimport shared_ptr, unique_ptr

from rapidsmpf._detail.exception_handling cimport ex_handler
from rapidsmpf.communicator.communicator cimport Communicator, cpp_Communicator
from rapidsmpf.memory.packed_data cimport cpp_PackedData
from rapidsmpf.streaming.core.context cimport Context, cpp_Context


cdef extern from "<rapidsmpf/streaming/coll/halo_exchange.hpp>" nogil:
    cdef cppclass cpp_HaloExchange "rapidsmpf::streaming::HaloExchange":
        cpp_HaloExchange(
            shared_ptr[cpp_Context] ctx,
            shared_ptr[cpp_Communicator] comm,
            int32_t op_id,
        ) except +ex_handler


cdef class HaloExchange:
    cdef unique_ptr[cpp_HaloExchange] _handle
    cdef Communicator _comm
    cdef Context _ctx
