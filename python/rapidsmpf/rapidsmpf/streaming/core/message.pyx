# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libcpp.utility cimport move


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
