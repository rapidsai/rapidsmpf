# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint8_t
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector

from rapidsmpf._detail.exception_handling cimport ex_handler
from rapidsmpf.streaming.core.channel cimport cpp_Channel
from rapidsmpf.streaming.core.context cimport cpp_Context
from rapidsmpf.streaming.core.node cimport cpp_Actor


cdef extern from "<rapidsmpf/streaming/core/fanout.hpp>" \
        namespace "rapidsmpf::streaming::actor" nogil:
    cpdef enum class FanoutPolicy (uint8_t):
        BOUNDED
        UNBOUNDED

    cdef cpp_Actor cpp_fanout \
        "rapidsmpf::streaming::actor::fanout"(
            shared_ptr[cpp_Context] ctx,
            shared_ptr[cpp_Channel] ch_in,
            vector[shared_ptr[cpp_Channel]] chs_out,
            FanoutPolicy policy
        ) except +ex_handler
