# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cpython.object cimport PyObject
from cpython.ref cimport Py_XDECREF
from cython.operator cimport dereference as deref
from libc.stddef cimport size_t
from libc.stdint cimport uint64_t
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.table.table_view cimport table_view as cpp_table_view
from pylibcudf.table cimport Table

from rapidsmpf.streaming.core.message cimport Message, cpp_Message


cdef extern from "<rapidsmpf/streaming/cudf/table_chunk.hpp>" nogil:
    cpp_Message cpp_to_message"rapidsmpf::streaming::to_message"\
        (uint64_t sequence_number, cpp_TableChunk&) except +


# Helper function to release a table chunk from a message, which is needed
# because TableChunk doesn't have a default ctor.
cdef extern from *:
    """
    namespace {
    std::unique_ptr<rapidsmpf::streaming::TableChunk>
    cpp_release_table_chunk_from_message(
        rapidsmpf::streaming::Message &&msg
    ) {
        return std::make_unique<rapidsmpf::streaming::TableChunk>(
            msg.release<rapidsmpf::streaming::TableChunk>()
        );
    }

    std::unique_ptr<rapidsmpf::streaming::TableChunk> cpp_from_table_view_with_owner(
        cudf::table_view view,
        std::size_t device_alloc_size,
        rmm::cuda_stream_view stream,
        PyObject *owner,
        void(*py_deleter)(void *),
        bool exclusive_view
    ) {
        // Called holding the gil.
        // Decref is done by the deleter.
        Py_XINCREF(owner);
        return std::make_unique<rapidsmpf::streaming::TableChunk>(
            view,
            device_alloc_size,
            stream,
            rapidsmpf::streaming::OwningWrapper(owner, py_deleter),
            exclusive_view ?
                rapidsmpf::streaming::TableChunk::ExclusiveView::YES
                : rapidsmpf::streaming::TableChunk::ExclusiveView::NO
        );
    }
    }
    """
    unique_ptr[cpp_TableChunk] \
        cpp_release_table_chunk_from_message(cpp_Message) except +

    unique_ptr[cpp_TableChunk] cpp_from_table_view_with_owner(...) except +


cdef void py_deleter(void *p) noexcept nogil:
    if p != NULL:
        with gil:
            Py_XDECREF(<PyObject*>p)


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
    cdef TableChunk from_handle(unique_ptr[cpp_TableChunk] handle):
        """
        Construct a TableChunk from an existing C++ handle.

        Parameters
        ----------
        handle
            A unique pointer to a C++ TableChunk.

        Returns
        -------
        A new TableChunk wrapping the given handle.
        """
        cdef TableChunk ret = TableChunk.__new__(TableChunk)
        ret._handle = move(handle)
        return ret

    @staticmethod
    def from_pylibcudf_table(
        Table table not None,
        Stream stream not None,
        *,
        bool_t exclusive_view,
    ):
        """
        Construct a TableChunk from a pylibcudf Table.

        Parameters
        ----------
        table
            A pylibcudf Table to wrap as a TableChunk.
        stream
            The CUDA stream on which this chunk was created.
        exclusive_view
            Indicates that this TableChunk has exclusive ownership semantics for the
            underlying table view.

            When ``True``, the following guarantees must hold:
              - The pylibcudf Table is the sole representation of the table data,
                i.e. no views exist.
              - The Table object exclusively owns the table's device memory.

            These guarantees allow the TableChunk to be spillable and ensure that
            when the owner is destroyed, the underlying device memory is correctly
            freed.

        Returns
        -------
        TableChunk
            A new TableChunk wrapping the given pylibcudf Table.

        Notes
        -----
        The returned TableChunk maintains a reference to ``table`` to ensure
        its underlying buffers remain valid for the lifetime of the chunk.
        This reference is managed by the underlying C++ object, so it
        persists even when the chunk is transferred through Channels.

        Warning
        -------
        This object does not keep the provided stream alive. The caller must
        ensure the stream remains valid for the lifetime of the streaming pipeline.
        """
        cdef cuda_stream_view _stream = stream.view()
        cdef size_t device_alloc_size = 0
        for col in table.columns():
            device_alloc_size += (<Column?>col).device_buffer_size()

        cdef cpp_table_view view = table.view()
        return TableChunk.from_handle(
            cpp_from_table_view_with_owner(
                view,
                device_alloc_size,
                _stream,
                <PyObject *>table,
                py_deleter,
                exclusive_view,
            )
        )

    @staticmethod
    def from_message(Message message not None):
        """
        Construct a TableChunk by consuming a Message.

        Parameters
        ----------
        message
            Message containing a TableChunk. The message is released and is empty
            after this call.

        Returns
        -------
        A new TableChunk extracted from the given message.
        """
        return TableChunk.from_handle(
            cpp_release_table_chunk_from_message(move(message._handle))
        )

    def into_message(self, uint64_t sequence_number, Message message not None):
        """
        Move this TableChunk into an empty Message.

        This method is not typically called directly. Instead, it is invoked by
        `Message.__init__()` when creating a new Message with this TableChunk as
        its payload.

        Parameters
        ----------
        sequence_number
            Ordering identifier for the message.
        message
            Message object that will take ownership of this TableChunk.

        Raises
        ------
        ValueError
            If the provided message is not empty.

        Warnings
        --------
        The TableChunk is released and must not be used after this call.
        """
        if not message.empty():
            raise ValueError("cannot move into a non-empty message")
        message._handle = cpp_to_message(
            sequence_number, move(deref(self.release_handle()))
        )

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
            raise ValueError("TableChunk is uninitialized, has it been released?")
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
            raise ValueError("TableChunk is uninitialized, has it been released?")
        return move(self._handle)

    @property
    def stream(self):
        """
        Return the CUDA stream on which this chunk was created.

        Returns
        -------
        Stream
            The CUDA stream.
        """
        return Stream._from_cudaStream_t(
            deref(self.handle_ptr()).stream().value()
        )

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
        return Table.from_table_view_of_arbitrary(ret, owner=self, stream=self.stream)

    def is_spillable(self):
        """
        Indicates whether this chunk can be spilled.

        A chunk is considered spillable if it was created from one of the following:
          - A message (via ``.from_message()``).
          - An exclusive pylibcudf table (via
            ``.from_pylibcudf_table(..., exclusive_view=True)``).

        Both of these creation paths imply device-owning semantics, meaning the
        TableChunk owns its underlying memory and can safely be spilled to host memory.

        Returns
        -------
        True if the table chunk can be spilled, otherwise, False.
        """
        return deref(self.handle_ptr()).is_spillable()
