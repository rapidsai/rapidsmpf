# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference as deref
from cython.operator cimport preincrement
from libc.stdint cimport uint64_t
from libcpp.map cimport map as cpp_map
from libcpp.memory cimport make_unique
from libcpp.utility cimport move

from rapidsmpf.buffer.content_description cimport content_description_from_cpp
from rapidsmpf.buffer.resource cimport BufferResource


cdef class SpillableMessages:
    def __init__(self):
        SpillableMessages.from_handle(make_unique[cpp_SpillableMessages]())

    def __dealloc__(self):
        with nogil:
            self._handle.reset()

    @staticmethod
    cdef from_handle(unique_ptr[cpp_SpillableMessages] handle):
        cdef SpillableMessages ret = SpillableMessages.__new__(SpillableMessages)
        ret._handle = move(handle)
        return ret

    def insert(self, Message message not None):
        cdef uint64_t ret
        with nogil:
            ret = deref(self._handle).insert(move(message._handle))
        return ret

    def extract(self, uint64_t mid):
        cdef cpp_Message ret
        with nogil:
            ret = deref(self._handle).extract(mid)
        return Message.from_handle(move(ret))

    def spill(self, uint64_t mid, BufferResource br not None):
        cdef cpp_BufferResource* _br = br.ptr()
        cdef size_t ret
        with nogil:
            ret = deref(self._handle).spill(mid, _br)
        return ret

    def get_content_descriptions(self):
        cdef cpp_map[cpp_SpillableMessages.cpp_MessageId, cpp_ContentDescription] cds
        with nogil:
            cds = deref(self._handle).get_content_descriptions()
        cdef cpp_map[
            cpp_SpillableMessages.cpp_MessageId, cpp_ContentDescription
        ].iterator it = cds.begin()

        ret = {}
        while it != cds.end():
            ret[deref(it).first] = content_description_from_cpp(deref(it).second)
            preincrement(it)
        return ret
