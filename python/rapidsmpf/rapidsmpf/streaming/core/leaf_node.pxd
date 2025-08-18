# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.vector cimport vector

from rapidsmpf.streaming.core.channel cimport cpp_SharedChannel
from rapidsmpf.streaming.core.context cimport cpp_Context
from rapidsmpf.streaming.core.node cimport cpp_Node


cdef extern from "<rapidsmpf/streaming/core/leaf_node.hpp>" nogil:
    cdef cpp_Node cpp_push_chunks_to_channel \
        "rapidsmpf::streaming::node::push_chunks_to_channel"[T](
            shared_ptr[cpp_Context] ctx,
            cpp_SharedChannel[T] ch_out,
            vector[unique_ptr[T]] chunks,
        ) except +
