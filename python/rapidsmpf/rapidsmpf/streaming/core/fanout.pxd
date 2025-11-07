# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint8_t
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector

from rapidsmpf.streaming.core.channel cimport cpp_Channel
from rapidsmpf.streaming.core.context cimport cpp_Context
from rapidsmpf.streaming.core.node cimport cpp_Node


cdef extern from "<rapidsmpf/streaming/core/fanout.hpp>" \
        namespace "rapidsmpf::streaming::node" nogil:
    cdef enum class cpp_FanoutPolicy \
            "rapidsmpf::streaming::node::FanoutPolicy" (uint8_t):
        BOUNDED "rapidsmpf::streaming::node::FanoutPolicy::BOUNDED"
        UNBOUNDED "rapidsmpf::streaming::node::FanoutPolicy::UNBOUNDED"

    cdef cpp_Node cpp_fanout \
        "rapidsmpf::streaming::node::fanout"(
            shared_ptr[cpp_Context] ctx,
            shared_ptr[cpp_Channel] ch_in,
            vector[shared_ptr[cpp_Channel]] chs_out,
            cpp_FanoutPolicy policy
        ) except +
