# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference as deref
from libc.stddef cimport size_t
from libc.stdint cimport uint64_t
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.table.table_view cimport table_view as cpp_table_view
from pylibcudf.table cimport Table
from rmm.librmm.cuda_stream_view cimport cuda_stream_view

from rapidsmpf.streaming.core.channel cimport Message, cpp_Message


# Help function to release a table chunk from a message, which is needed
# because TableChunk doesn't have a default ctor.
cdef extern from *:
    """
    std::unique_ptr<rapidsmpf::streaming::TableChunk>
    cpp_release_table_chunk_from_message(
        rapidsmpf::streaming::Message &&msg
    ) {
        return std::make_unique<rapidsmpf::streaming::TableChunk>(
            msg.release<rapidsmpf::streaming::TableChunk>()
        );
    }
    """
    unique_ptr[cpp_TableChunk] \
        cpp_release_table_chunk_from_message(cpp_Message) except +


cdef class TableChunk:
    """
    A unit of table data in a streaming pipeline.

    Represents either an unpacked pylibcudf table, a packed (serialized) table,
    or `rapidsmpf.buffer.packed_data.PackedData`.

    A TableChunk may be initially unavailable (e.g., if the data is packed or
    spilled), and can be made available (i.e., materialized to device memory)
    on demand.

    Use the factory functions `from_pylibcudf_table` and `from_message` to
    create a new table chunk.
    """
    def __init__(self):
        raise ValueError("use the `from_*` factory functions")

    def __dealloc__(self):
        with nogil:
            self._handle.reset()

    @staticmethod
    cdef TableChunk from_handle(
        unique_ptr[cpp_TableChunk] handle, Stream stream, object owner
    ):
        """
        Construct a TableChunk from an existing C++ handle.

        Parameters
        ----------
        handle
            A unique pointer to a C++ TableChunk.
        stream
            The CUDA stream on which this chunk was created. If `None`,
            the stream is obtained from the handle.
        owner
            Python object that owns the underlying buffers and must
            be kept alive for the lifetime of this TableChunk.

        Returns
        -------
        A new TableChunk wrapping the given handle.
        """

        if stream is None:
            stream = Stream._from_cudaStream_t(
                deref(handle).stream().value()
            )
        cdef TableChunk ret = TableChunk.__new__(TableChunk)
        ret._handle = move(handle)
        ret._stream = stream
        ret._owner = owner
        return ret

    @staticmethod
    def from_pylibcudf_table(uint64_t sequence_number, Table table, stream):
        """
        Construct a TableChunk from a pylibcudf Table.

        Parameters
        ----------
        sequence_number
            Sequence number of this new chunk.
        table
            A pylibcudf Table to wrap as a TableChunk.
        stream
            The CUDA stream on which this chunk was created.

        Returns
        -------
        A new TableChunk wrapping the given pylibcudf Table.

        Notes
        -----
        The returned TableChunk maintains a reference to `table` to ensure
        its underlying buffers remain valid for the lifetime of the chunk.
        """
        if stream is None:
            raise ValueError("stream cannot be None")
        cdef Stream _stream = Stream(stream)
        cdef cuda_stream_view _stream_view = _stream.view()

        cdef size_t device_alloc_size = 0
        for col in table.columns():
            device_alloc_size += (<Column?>col).device_buffer_size()

        cdef cpp_table_view view = table.view()
        cdef unique_ptr[cpp_TableChunk] ret
        with nogil:
            ret = make_unique[cpp_TableChunk](
                sequence_number,
                view,
                device_alloc_size,
                _stream_view
            )
        return TableChunk.from_handle(move(ret), stream=_stream, owner=table)

    @staticmethod
    def from_message(Message message):
        """
        Construct a TableChunk by consuming a Message.

        Parameters
        ----------
        message
            Message containing a TableChunk. The message is consumed
            and is empty after this call.

        Returns
        -------
        A new TableChunk extracted from the given message.
        """
        return TableChunk.from_handle(
            cpp_release_table_chunk_from_message(move(message._handle)),
            stream = None,
            owner = None,
        )

    def into_message(self, Message message):
        """
        Move this TableChunk into an empty Message.

        This method is not typically called directly. Instead, it is invoked by
        `Message.__init__()` when creating a new Message with this TableChunk as
        its payload.

        Parameters
        ----------
        message
            Message object that will take ownership of this TableChunk.

        Raises
        ------
        ValueError
            If the provided message is not empty.

        Notes
        -----
        This operation consumes the TableChunk. After the call, the current
        object is in a moved-from state and must not be accessed.
        """
        if not message.empty():
            raise ValueError("cannot move into a non-empty message")
        message._handle = cpp_Message(self.release_handle())

    cdef const cpp_TableChunk* handle_ptr(self):
        """
        Return a pointer to the underlying C++ TableChunk.

        Returns
        -------
        Raw pointer to the underlying C++ object.

        Raises
        ------
        ValueError
            If the TableChunk is uninitialized.
        """
        if not self._handle:
            raise ValueError("TableChunk is uninitialized, has it been consumed?")
        return self._handle.get()

    cdef unique_ptr[cpp_TableChunk] release_handle(self):
        """
        Release ownership of the underlying C++ TableChunk.

        After this call, the current object is in a moved-from state and
        must not be accessed.

        Returns
        -------
        Unique pointer to the underlying C++ object.

        Raises
        ------
        ValueError
            If the TableChunk is uninitialized.
        """
        if not self._handle:
            raise ValueError("TableChunk is uninitialized, has it been consumed?")
        return move(self._handle)

    def sequence_number(self):
        """
        Return the sequence number of this chunk.

        Returns
        -------
        The sequence number.
        """
        return deref(self.handle_ptr()).sequence_number()

    def stream(self):
        """
        Return the CUDA stream on which this chunk was created.

        Returns
        -------
        Stream
            The CUDA stream.
        """
        return self._stream

    def data_alloc_size(self, MemoryType mem_type):
        """
        Number of bytes allocated for the data in the specified memory type.

        Parameters
        ----------
        mem_type
            The memory type to query.

        Returns
        -------
        Number of bytes allocated.
        """
        return deref(self.handle_ptr()).data_alloc_size(mem_type)

    def is_available(self):
        """
        Indicates whether the underlying table data is fully available in
        device memory.

        Returns
        -------
        True if the table is already available; otherwise, False.
        """
        return deref(self.handle_ptr()).is_available()

    def table_view(self):
        """
        Returns a view of the underlying pylibcudf table.

        The table must be available in device memory.

        Returns
        -------
        A view of the underlying table. The view holds a reference to this
        `TableChunk` to ensure it remains alive.

        Raises
        ------
        ValueError
            If ``self.is_available() is False``.
        """
        cdef const cpp_TableChunk* handle = self.handle_ptr()
        cdef cpp_table_view ret
        with nogil:
            ret = deref(handle).table_view()
        return Table.from_table_view_of_arbitrary(ret, owner=self)
