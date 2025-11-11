# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference as deref
from libcpp.utility cimport move

from rapidsmpf.buffer.content_description cimport content_description_from_cpp
from rapidsmpf.buffer.resource cimport MemoryReservation


cdef class Message:
    """
    A message to be transferred between streaming nodes.

    Parameters
    ----------
    sequence_number
        Ordering identifier for the message.
    payload
        A payload object that implements the `Payload` protocol. The payload is
        moved into this message.

    Warnings
    --------
    `payload` is released by this call and must not be used afterwards.
    """
    def __init__(self, uint64_t sequence_number, payload):
        payload.into_message(sequence_number, self)

    @staticmethod
    cdef from_handle(cpp_Message handle):
        """
        Construct a Message from an existing C++ handle.

        Parameters
        ----------
        handle
            A C++ message handle whose ownership will be **moved** into the
            returned `Message`.

        Returns
        -------
        A new Python `Message` object owning `handle`.
        """
        cdef Message ret = Message.__new__(Message)
        ret._handle = move(handle)
        return ret

    def __dealloc__(self):
        with nogil:
            self._handle.reset()

    @classmethod
    def __class_getitem__(cls, item):
        return cls

    def empty(self):
        """
        Return whether this message is empty.

        Returns
        -------
        True if the message is empty; otherwise, False.
        """
        cdef bool_t ret
        with nogil:
            ret = self._handle.empty()
        return ret

    @property
    def sequence_number(self):
        """
        Return the sequence number of this message.

        Returns
        -------
        The sequence number.
        """
        cdef uint64_t ret
        with nogil:
            ret = self._handle.sequence_number()
        return ret

    def get_content_description(self):
        return content_description_from_cpp(self._handle.content_description())

    def copy_cost(self):
        cdef uint64_t ret
        with nogil:
            ret = self._handle.copy_cost()
        return ret

    def copy(self, MemoryReservation reservation not None):
        cdef cpp_Message ret
        cdef cpp_MemoryReservation* res = reservation._handle.get()
        with nogil:
            ret = self._handle.copy(deref(res))
        return Message.from_handle(move(ret))
