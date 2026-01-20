# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cpython.object cimport PyObject
from cpython.ref cimport Py_DECREF, Py_INCREF
from cython.operator cimport dereference as deref
from libc.stdint cimport uint64_t
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.utility cimport move

from rapidsmpf.owning_wrapper cimport cpp_OwningWrapper
from rapidsmpf.streaming.chunks.utils cimport py_deleter
from rapidsmpf.streaming.core.message cimport Message, cpp_Message


cdef extern from * nogil:
    """
    namespace {
    rapidsmpf::streaming::Message cpp_to_message(
        std::uint64_t sequence_number,
        std::unique_ptr<rapidsmpf::OwningWrapper> obj
    ) {
        return rapidsmpf::streaming::Message{
            sequence_number,
            std::move(obj),
            // TODO: support content description in Python. For now, we use
            // an empty content description and provide no copy callback. This
            // means that ArbitraryChunk won't support copy or spilling.
            rapidsmpf::ContentDescription{}
        };
    }
    }  // namespace
    """
    cpp_Message cpp_to_message(
        uint64_t, unique_ptr[cpp_OwningWrapper]
    ) except +


cdef class ArbitraryChunk:
    """
    A chunk containing an arbitrary Python object as a payload.

    Parameters
    ----------
    obj
        The payload object.

    Notes
    -----
    To extract the object from the chunk, use :meth:`release`. The object
    is stored in a unique pointer with a custom deleter, so it is safe to
    drop the chunk in C++: deallocation will acquire the gil and decref the
    stored object.
    """
    def __init__(self, object obj):
        Py_INCREF(obj)
        self._handle = make_unique[cpp_OwningWrapper](
            <void *><PyObject *>obj, py_deleter
        )

    @classmethod
    def __class_getitem__(cls, args):
        return cls

    def __dealloc__(self):
        with nogil:
            self._handle.reset()

    @staticmethod
    cdef ArbitraryChunk from_handle(unique_ptr[cpp_OwningWrapper] handle):
        """
        Construct an ArbitraryChunk from an existing C++ handle.

        Parameters
        ----------
        handle
            A unique pointer to a C++ OwningWrapper.

        Returns
        -------
        A new ArbitraryChunk wrapping the given handle.
        """

        cdef ArbitraryChunk ret = ArbitraryChunk.__new__(ArbitraryChunk)
        ret._handle = move(handle)
        return ret

    def release(self):
        cdef unique_ptr[cpp_OwningWrapper] obj = self.release_handle()
        cdef object pyobj = <object><PyObject *>(deref(obj).release())
        # Cast to object increfs, so we must decref here
        Py_DECREF(pyobj)
        return pyobj

    @staticmethod
    def from_message(Message message not None):
        """
        Construct an ArbitraryChunk by consuming a Message.

        Parameters
        ----------
        message
            Message containing an ArbitraryChunk. The message is released
            and is empty after this call.

        Returns
        -------
        A new ArbitraryChunk extracted from the given message.
        """
        return ArbitraryChunk.from_handle(
            make_unique[cpp_OwningWrapper](
                message._handle.release[cpp_OwningWrapper]()
            )
        )

    def into_message(self, uint64_t sequence_number, Message message not None):
        """
        Move this ArbitraryChunk into a Message.

        This method is not typically called directly. Instead, it is invoked by
        `Message.__init__()` when creating a new Message with this ArbitraryChunk
        as its payload.

        Parameters
        ----------
        sequence_number
            Ordering identifier for the message.
        message
            Message object that will take ownership of this ArbitraryChunk.

        Raises
        ------
        ValueError
            If the provided message is not empty.

        Warnings
        --------
        The ArbitraryChunk is released and must not be used after this call.
        """
        if not message.empty():
            raise ValueError("cannot move into a non-empty message")
        message._handle = cpp_to_message(
            sequence_number, move(self.release_handle())
        )

    cdef const cpp_OwningWrapper* handle_ptr(self):
        """
        Return a pointer to the underlying C++ OwningWrapper.

        Returns
        -------
        Raw pointer to the underlying C++ object.

        Raises
        ------
        ValueError
            If the ArbitraryChunk is uninitialized.
        """
        if not self._handle:
            raise ValueError("is uninitialized, has it been released?")
        return self._handle.get()

    cdef unique_ptr[cpp_OwningWrapper] release_handle(self):
        """
        Release ownership of the underlying C++ OwningWrapper.

        After this call, the current object is in a moved-from state and
        must not be accessed.

        Returns
        -------
        Unique pointer to the underlying C++ object.

        Raises
        ------
        ValueError
            If the OwningWrapperChunk is uninitialized.
        """
        if not self._handle:
            raise ValueError("is uninitialized, has it been released?")
        return move(self._handle)
