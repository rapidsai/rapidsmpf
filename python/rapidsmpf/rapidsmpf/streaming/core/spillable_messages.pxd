# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libc.stddef cimport size_t
from libc.stdint cimport uint64_t
from libcpp.map cimport map as cpp_map
from libcpp.memory cimport shared_ptr

from rapidsmpf.memory.buffer_resource cimport cpp_BufferResource
from rapidsmpf.memory.content_description cimport cpp_ContentDescription
from rapidsmpf.streaming.core.message cimport Message, cpp_Message


cdef extern from "<rapidsmpf/streaming/core/spillable_messages.hpp>" nogil:
    cdef cppclass cpp_SpillableMessages"rapidsmpf::streaming::SpillableMessages":
        ctypedef uint64_t cpp_MessageId"MessageId"

        cpp_MessageId insert(cpp_Message message) except +
        cpp_Message extract(cpp_MessageId mid) except +
        size_t spill(cpp_MessageId mid, cpp_BufferResource *br) except +
        cpp_map[cpp_MessageId, cpp_ContentDescription] \
            get_content_descriptions() except +


cdef class SpillableMessages:
    cdef shared_ptr[cpp_SpillableMessages] _handle

    @staticmethod
    cdef from_handle(shared_ptr[cpp_SpillableMessages] handle)
