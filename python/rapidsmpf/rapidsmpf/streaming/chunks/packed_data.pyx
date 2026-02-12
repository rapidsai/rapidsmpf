# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint64_t
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from rapidsmpf._detail.exception_handling cimport ex_handler
from rapidsmpf.memory.packed_data cimport PackedData, cpp_PackedData
from rapidsmpf.streaming.core.message cimport Message, cpp_Message


cdef extern from "<rapidsmpf/streaming/chunks/packed_data.hpp>" nogil:
    cpp_Message cpp_to_message"rapidsmpf::streaming::to_message"\
        (uint64_t sequence_number, unique_ptr[cpp_PackedData]) except +ex_handler

cdef extern from * nogil:
    """
    namespace {
    std::unique_ptr<rapidsmpf::PackedData>
    cpp_from_message(rapidsmpf::streaming::Message msg) {
        return std::make_unique<rapidsmpf::PackedData>(
            msg.release<rapidsmpf::PackedData>()
        );
    }
    }  // namespace
    """
    unique_ptr[cpp_PackedData] cpp_from_message(cpp_Message) except +ex_handler


cdef class PackedDataChunk:
    def __init__(self):
        raise ValueError("use the `from_*` factory functions")

    def __dealloc__(self):
        with nogil:
            self._handle.reset()

    def to_packed_data(self):
        """
        Convert to a PackedData object.

        Returns
        -------
        A new PackedData from this chunk. The chunk is left empty.
        """
        return PackedData.from_librapidsmpf(self.release_handle())

    @staticmethod
    def from_packed_data(PackedData obj not None):
        """
        Construct a PackedDataChunk from an existing PackedData object.

        Parameters
        ----------
        obj
            The PackedData to construct from. The packed data is empty after this call.

        Returns
        -------
        A new PackedDataChunk from the given object.
        """
        return PackedDataChunk.from_handle(move(obj.c_obj))

    @staticmethod
    cdef PackedDataChunk from_handle(
        unique_ptr[cpp_PackedData] handle
    ):
        """
        Construct a PackedDataChunk from an existing C++ handle.

        Parameters
        ----------
        handle
            A unique pointer to a C++ PackedDataChunk.

        Returns
        -------
        A new PackedDataChunk wrapping the given handle.
        """

        cdef PackedDataChunk ret = PackedDataChunk.__new__(PackedDataChunk)
        ret._handle = move(handle)
        return ret

    @staticmethod
    def from_message(Message message not None):
        """
        Construct a PackedDataChunk by consuming a Message.

        Parameters
        ----------
        message
            Message containing a PackedDataChunk. The message is released
            and is empty after this call.

        Returns
        -------
        A new PackedDataChunk extracted from the given message.
        """
        return PackedDataChunk.from_handle(
            cpp_from_message(move(message._handle))
        )

    def into_message(self, uint64_t sequence_number, Message message not None):
        """
        Move this PackedDataChunk into a Message.

        This method is not typically called directly. Instead, it is invoked by
        `Message.__init__()` when creating a new Message with this PackedDataChunk
        as its payload.

        Parameters
        ----------
        sequence_number
            Ordering identifier for the message.
        message
            Message object that will take ownership of this PackedDataChunk.

        Raises
        ------
        ValueError
            If the provided message is not empty.

        Warnings
        --------
        The PackedDataChunk is released and must not be used after this call.
        """
        if not message.empty():
            raise ValueError("cannot move into a non-empty message")
        message._handle = cpp_to_message(
            sequence_number, move(self.release_handle())
        )

    cdef const cpp_PackedData* handle_ptr(self):
        """
        Return a pointer to the underlying C++ PackedDataChunk.

        Returns
        -------
        Raw pointer to the underlying C++ object.

        Raises
        ------
        ValueError
            If the PackedDataChunk is uninitialized.
        """
        if not self._handle:
            raise ValueError("is uninitialized, has it been released?")
        return self._handle.get()

    cdef unique_ptr[cpp_PackedData] release_handle(self):
        """
        Release ownership of the underlying C++ PackedDataChunk.

        After this call, the current object is in a moved-from state and
        must not be accessed.

        Returns
        -------
        Unique pointer to the underlying C++ object.

        Raises
        ------
        ValueError
            If the PackedDataChunk is uninitialized.
        """
        if not self._handle:
            raise ValueError("is uninitialized, has it been released?")
        return move(self._handle)
