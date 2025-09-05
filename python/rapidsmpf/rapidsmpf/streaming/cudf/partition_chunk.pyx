# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference as deref
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.utility cimport move
from rmm.pylibrmm.stream cimport Stream

from rapidsmpf.streaming.core.channel cimport Message, cpp_Message


cdef class PartitionMapChunk:
    def __init__(self):
        raise ValueError("use the `from_*` factory functions")

    def __dealloc__(self):
        with nogil:
            self._handle.reset()

    @staticmethod
    cdef PartitionMapChunk from_handle(
        unique_ptr[cpp_PartitionMapChunk] handle, Stream stream, object owner
    ):
        """
        Construct a PartitionMapChunk from an existing C++ handle.

        Parameters
        ----------
        handle
            A unique pointer to a C++ PartitionMapChunk.
        stream
            The CUDA stream on which this chunk was created. If `None`,
            the stream is obtained from the handle.
        owner
            Python object that owns the underlying buffers and must
            be kept alive for the lifetime of this PartitionMapChunk.

        Returns
        -------
        A new PartitionMapChunk wrapping the given handle.
        """

        if stream is None:
            stream = Stream._from_cudaStream_t(
                deref(handle).stream.value()
            )
        cdef PartitionMapChunk ret = PartitionMapChunk.__new__(PartitionMapChunk)
        ret._handle = move(handle)
        ret._stream = stream
        ret._owner = owner
        return ret

    @staticmethod
    def from_message(Message message):
        """
        Construct a PartitionMapChunk by consuming a Message.

        Parameters
        ----------
        message
            Message containing a PartitionMapChunk. The message is released
            and is empty after this call.

        Returns
        -------
        A new PartitionMapChunk extracted from the given message.
        """
        return PartitionMapChunk.from_handle(
            make_unique[cpp_PartitionMapChunk](
                message._handle.release[cpp_PartitionMapChunk]()
            ),
            stream = None,
            owner = None,
        )

    def into_message(self, Message message):
        """
        Move this PartitionMapChunk into a Message.

        This method is not typically called directly. Instead, it is invoked by
        `Message.__init__()` when creating a new Message with this PartitionMapChunk
        as its payload.

        Parameters
        ----------
        message
            Message object that will take ownership of this PartitionMapChunk.

        Raises
        ------
        ValueError
            If the provided message is not empty.

        Warnings
        --------
        The PartitionMapChunk is released and must not be used after this call.
        """
        message._handle = cpp_Message(self.release_handle())

    cdef const cpp_PartitionMapChunk* handle_ptr(self):
        """
        Return a pointer to the underlying C++ PartitionMapChunk.

        Returns
        -------
        Raw pointer to the underlying C++ object.

        Raises
        ------
        ValueError
            If the PartitionMapChunk is uninitialized.
        """
        if not self._handle:
            raise ValueError("is uninitialized, has it been released?")
        return self._handle.get()

    cdef unique_ptr[cpp_PartitionMapChunk] release_handle(self):
        """
        Release ownership of the underlying C++ PartitionMapChunk.

        After this call, the current object is in a moved-from state and
        must not be accessed.

        Returns
        -------
        Unique pointer to the underlying C++ object.

        Raises
        ------
        ValueError
            If the PartitionMapChunk is uninitialized.
        """
        if not self._handle:
            raise ValueError("is uninitialized, has it been released?")
        return move(self._handle)

    @property
    def sequence_number(self):
        """
        Return the sequence number of this chunk.

        Returns
        -------
        The sequence number.
        """
        return deref(self.handle_ptr()).sequence_number

    @property
    def stream(self):
        """
        Return the CUDA stream on which this chunk was created.

        Returns
        -------
        Stream
            The CUDA stream.
        """
        return self._stream


cdef class PartitionVectorChunk:
    def __init__(self):
        raise ValueError("use the `from_*` factory functions")

    def __dealloc__(self):
        with nogil:
            self._handle.reset()

    @staticmethod
    cdef PartitionVectorChunk from_handle(
        unique_ptr[cpp_PartitionVectorChunk] handle, Stream stream, object owner
    ):
        """
        Construct a PartitionVectorChunk from an existing C++ handle.

        Parameters
        ----------
        handle
            A unique pointer to a C++ PartitionVectorChunk.
        stream
            The CUDA stream on which this chunk was created. If `None`,
            the stream is obtained from the handle.
        owner
            Python object that owns the underlying buffers and must
            be kept alive for the lifetime of this PartitionVectorChunk.

        Returns
        -------
        A new PartitionVectorChunk wrapping the given handle.
        """

        if stream is None:
            stream = Stream._from_cudaStream_t(
                deref(handle).stream.value()
            )
        cdef PartitionVectorChunk ret = PartitionVectorChunk.__new__(
            PartitionVectorChunk
        )
        ret._handle = move(handle)
        ret._stream = stream
        ret._owner = owner
        return ret

    @staticmethod
    def from_message(Message message):
        """
        Construct a PartitionVectorChunk by consuming a Message.

        Parameters
        ----------
        message
            Message containing a PartitionVectorChunk. The message is released
            and is empty after this call.

        Returns
        -------
        A new PartitionVectorChunk extracted from the given message.
        """
        return PartitionVectorChunk.from_handle(
            make_unique[cpp_PartitionVectorChunk](
                message._handle.release[cpp_PartitionVectorChunk]()
            ),
            stream = None,
            owner = None,
        )

    def into_message(self, Message message):
        """
        Move this PartitionVectorChunk into a Message.

        This method is not typically called directly. Instead, it is invoked by
        `Message.__init__()` when creating a new Message with this PartitionVectorChunk
        as its payload.

        Parameters
        ----------
        message
            Message object that will take ownership of this PartitionVectorChunk.

        Raises
        ------
        ValueError
            If the provided message is not empty.

        Warnings
        --------
        The PartitionVectorChunk is released and must not be used after this call.
        """
        message._handle = cpp_Message(self.release_handle())

    cdef const cpp_PartitionVectorChunk* handle_ptr(self):
        """
        Return a pointer to the underlying C++ PartitionVectorChunk.

        Returns
        -------
        Raw pointer to the underlying C++ object.

        Raises
        ------
        ValueError
            If the PartitionVectorChunk is uninitialized.
        """
        if not self._handle:
            raise ValueError("is uninitialized, has it been released?")
        return self._handle.get()

    cdef unique_ptr[cpp_PartitionVectorChunk] release_handle(self):
        """
        Release ownership of the underlying C++ PartitionVectorChunk.

        After this call, the current object is in a moved-from state and
        must not be accessed.

        Returns
        -------
        Unique pointer to the underlying C++ object.

        Raises
        ------
        ValueError
            If the PartitionVectorChunk is uninitialized.
        """
        if not self._handle:
            raise ValueError("is uninitialized, has it been released?")
        return move(self._handle)

    def sequence_number(self):
        """
        Return the sequence number of this chunk.

        Returns
        -------
        The sequence number.
        """
        return deref(self.handle_ptr()).sequence_number

    def stream(self):
        """
        Return the CUDA stream on which this chunk was created.

        Returns
        -------
        Stream
            The CUDA stream.
        """
        return self._stream
