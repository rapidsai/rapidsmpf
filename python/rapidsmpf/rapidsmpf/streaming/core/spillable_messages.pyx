# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference as deref
from cython.operator cimport preincrement
from libc.stdint cimport uint64_t
from libcpp.map cimport map as cpp_map
from libcpp.memory cimport make_shared, shared_ptr
from libcpp.utility cimport move

from rapidsmpf.memory.buffer_resource cimport BufferResource
from rapidsmpf.memory.content_description cimport content_description_from_cpp


cdef class SpillableMessages:
    """
    Container for individually spillable messages.

    This class manages a collection of messages that can be spilled,
    extracted, or inspected independently. Each inserted message is assigned
    a unique identifier that can later be used to extract or spill it.
    The container is thread-safe for concurrent insertions, extractions,
    and spills.

    Examples
    --------
    >>> msgs = SpillableMessages()
    >>> mid = msgs.insert(msg)
    >>> msgs.spill(mid=mid, br=br)
    >>> recovered = msgs.extract(mid=mid)
    """
    def __init__(self):
        self._handle = make_shared[cpp_SpillableMessages]()

    def __dealloc__(self):
        with nogil:
            self._handle.reset()

    @staticmethod
    cdef from_handle(shared_ptr[cpp_SpillableMessages] handle):
        """Create a new instance from an existing C++ handle."""
        cdef SpillableMessages ret = SpillableMessages.__new__(SpillableMessages)
        ret._handle = move(handle)
        return ret

    def insert(self, Message message not None):
        """
        Insert a new message into the container.

        Parameters
        ----------
        message
            The message to insert.

        Returns
        -------
        The unique identifier assigned to the inserted message.
        """
        cdef uint64_t ret
        with nogil:
            ret = deref(self._handle).insert(move(message._handle))
        return ret

    def extract(self, *, uint64_t mid):
        """
        Extract a message by its identifier.

        Parameters
        ----------
        mid
            Identifier of the message to extract.

        Returns
        -------
        The extracted message instance.
        """
        cdef cpp_Message ret
        with nogil:
            ret = deref(self._handle).extract(mid)
        return Message.from_handle(move(ret))

    def spill(self, *, uint64_t mid, BufferResource br not None):
        """
        Spill a specific message to an external buffer resource.

        Parameters
        ----------
        mid
            Identifier of the message to spill.
        br
            Buffer resource used for spill allocations.

        Returns
        -------
        The number of bytes spilled.
        """
        cdef cpp_BufferResource* _br = br.ptr()
        cdef size_t ret
        with nogil:
            ret = deref(self._handle).spill(mid, _br)
        return ret

    def get_content_descriptions(self):
        """
        Retrieve content descriptions for all messages.

        Returns
        -------
        A dict from message identifiers to content descriptions.
        """
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
