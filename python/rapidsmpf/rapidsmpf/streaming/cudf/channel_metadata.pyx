# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Channel metadata types for streaming pipelines."""

from cuda.bindings.cyruntime cimport cudaStream_t
from cython.operator cimport dereference as deref
from libc.stdint cimport int32_t, uint64_t
from libcpp.memory cimport make_unique, shared_ptr, unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector
from pylibcudf.libcudf.table.table_view cimport table_view as cpp_table_view
from pylibcudf.libcudf.types cimport null_order as cpp_null_order
from pylibcudf.libcudf.types cimport order as cpp_order
from pylibcudf.table cimport Table
from rmm.pylibrmm.stream cimport Stream

from rapidsmpf.streaming.core.message cimport Message
from rapidsmpf.streaming.cudf.table_chunk cimport TableChunk, cpp_TableChunk


cdef extern from * nogil:
    """
    #include <rapidsmpf/streaming/cudf/table_chunk.hpp>
    #include <memory>
    namespace {
    // Convert unique_ptr to shared_ptr (takes ownership)
    std::shared_ptr<rapidsmpf::streaming::TableChunk>
    unique_to_shared(std::unique_ptr<rapidsmpf::streaming::TableChunk> ptr) {
        return std::shared_ptr<rapidsmpf::streaming::TableChunk>(ptr.release());
    }
    // Check if shared_ptr is non-null
    bool has_chunk(const std::shared_ptr<rapidsmpf::streaming::TableChunk>& ptr) {
        return ptr != nullptr;
    }
    // Get table_view from shared_ptr<TableChunk>
    cudf::table_view get_boundaries_view(
        const std::shared_ptr<rapidsmpf::streaming::TableChunk>& ptr
    ) {
        return ptr->table_view();
    }
    // Get cudaStream_t from shared_ptr<TableChunk>
    cudaStream_t get_boundaries_stream(
        const std::shared_ptr<rapidsmpf::streaming::TableChunk>& ptr
    ) {
        return ptr->stream().value();
    }
    }
    """
    shared_ptr[cpp_TableChunk] unique_to_shared(
        unique_ptr[cpp_TableChunk]
    ) noexcept
    bint has_chunk(const shared_ptr[cpp_TableChunk]&) noexcept
    cpp_table_view get_boundaries_view(const shared_ptr[cpp_TableChunk]&) except +
    cudaStream_t get_boundaries_stream(const shared_ptr[cpp_TableChunk]&) noexcept


cdef extern from * nogil:
    """
    namespace {
    std::unique_ptr<rapidsmpf::streaming::ChannelMetadata>
    cpp_channel_metadata_from_message(rapidsmpf::streaming::Message msg) {
        return std::make_unique<rapidsmpf::streaming::ChannelMetadata>(
            msg.release<rapidsmpf::streaming::ChannelMetadata>()
        );
    }
    }
    """
    unique_ptr[cpp_ChannelMetadata] cpp_channel_metadata_from_message(
        cpp_Message
    ) except +


cdef class HashScheme:
    """Hash partitioning scheme: rows distributed by hash(column_indices) % modulus."""

    def __init__(self, tuple column_indices, int modulus):
        cdef vector[int32_t] cols
        for c in column_indices:
            cols.push_back(<int32_t?>c)
        self._scheme = cpp_HashScheme(cols, modulus)

    @staticmethod
    cdef HashScheme from_cpp(cpp_HashScheme scheme):
        cdef HashScheme ret = HashScheme.__new__(HashScheme)
        ret._scheme = move(scheme)
        return ret

    @property
    def column_indices(self) -> tuple:
        return tuple(self._scheme.column_indices)

    @property
    def modulus(self) -> int:
        return self._scheme.modulus

    def __eq__(self, other):
        if not isinstance(other, HashScheme):
            return NotImplemented
        return self._scheme == (<HashScheme>other)._scheme

    def __repr__(self):
        return f"HashScheme({self.column_indices!r}, {self.modulus})"


cdef cpp_order _str_to_order(str s) except *:
    """Convert Python string to cudf::order enum."""
    if s == "ascending":
        return cpp_order.ASCENDING
    elif s == "descending":
        return cpp_order.DESCENDING
    else:
        raise ValueError(f"Invalid order: {s!r}, expected 'ascending' or 'descending'")


cdef str _order_to_str(cpp_order o):
    """Convert cudf::order enum to Python string."""
    if o == cpp_order.ASCENDING:
        return "ascending"
    else:
        return "descending"


cdef cpp_null_order _str_to_null_order(str s) except *:
    """Convert Python string to cudf::null_order enum."""
    if s == "first":
        return cpp_null_order.BEFORE
    elif s == "last":
        return cpp_null_order.AFTER
    else:
        raise ValueError(f"Invalid null_order: {s!r}, expected 'first' or 'last'")


cdef str _null_order_to_str(cpp_null_order o):
    """Convert cudf::null_order enum to Python string."""
    if o == cpp_null_order.BEFORE:
        return "first"
    else:
        return "last"


cdef class OrderScheme:
    """Order-based partitioning scheme for sorted/range-partitioned data.

    Data is partitioned by value ranges based on predetermined boundaries.
    For N partitions, there are N-1 boundary rows.
    """

    def __init__(
        self,
        tuple column_indices,
        tuple orders,
        tuple null_orders,
        TableChunk boundaries = None,
    ):
        cdef vector[int32_t] cols
        cdef vector[cpp_order] ords
        cdef vector[cpp_null_order] nulls

        if len(column_indices) != len(orders):
            raise ValueError("column_indices and orders must have the same length")
        if len(column_indices) != len(null_orders):
            raise ValueError("column_indices and null_orders must have the same length")

        for c in column_indices:
            cols.push_back(<int32_t?>c)
        for o in orders:
            ords.push_back(_str_to_order(o))
        for n in null_orders:
            nulls.push_back(_str_to_null_order(n))

        self._scheme.column_indices = move(cols)
        self._scheme.orders = move(ords)
        self._scheme.null_orders = move(nulls)

        if boundaries is not None:
            # Move the TableChunk's handle into a shared_ptr (consumes the TableChunk)
            self._scheme.boundaries = unique_to_shared(boundaries.release_handle())

    @staticmethod
    cdef OrderScheme from_cpp(cpp_OrderScheme scheme):
        cdef OrderScheme ret = OrderScheme.__new__(OrderScheme)
        ret._scheme = move(scheme)
        return ret

    @property
    def column_indices(self) -> tuple:
        return tuple(self._scheme.column_indices)

    @property
    def orders(self) -> tuple:
        return tuple(_order_to_str(o) for o in self._scheme.orders)

    @property
    def null_orders(self) -> tuple:
        return tuple(_null_order_to_str(o) for o in self._scheme.null_orders)

    @property
    def has_boundaries(self) -> bool:
        """Check if boundaries are set."""
        return has_chunk(self._scheme.boundaries)

    def get_boundaries_table(self):
        """Get the boundaries as a pylibcudf.Table, or None if not set.

        Returns
        -------
        pylibcudf.Table or None
            The boundaries table, or None if not set.
        """
        if not has_chunk(self._scheme.boundaries):
            return None
        cdef cpp_table_view view = get_boundaries_view(self._scheme.boundaries)
        cdef cudaStream_t stream = get_boundaries_stream(self._scheme.boundaries)
        # owner=self keeps the underlying TableChunk (via shared_ptr) alive
        return Table.from_table_view_of_arbitrary(
            view, owner=self, stream=Stream._from_cudaStream_t(stream)
        )

    def __eq__(self, other):
        if not isinstance(other, OrderScheme):
            return NotImplemented
        return self._scheme == (<OrderScheme>other)._scheme

    def __repr__(self):
        return (
            f"OrderScheme({self.column_indices!r}, {self.orders!r}, "
            f"{self.null_orders!r}, has_boundaries={self.has_boundaries})"
        )


cdef cpp_PartitioningSpec _to_spec(obj) except *:
    """Convert Python object to PartitioningSpec."""
    if obj is None:
        return cpp_PartitioningSpec.none()
    elif obj == "inherit":
        return cpp_PartitioningSpec.inherit()
    elif isinstance(obj, HashScheme):
        return cpp_PartitioningSpec.from_hash((<HashScheme>obj)._scheme)
    elif isinstance(obj, OrderScheme):
        return cpp_PartitioningSpec.from_order((<OrderScheme>obj)._scheme)
    else:
        raise TypeError(
            f"Expected HashScheme, OrderScheme, None, or 'inherit', "
            f"got {type(obj).__name__}"
        )


cdef object _from_spec(cpp_PartitioningSpec spec):
    """Convert PartitioningSpec to Python object."""
    if spec.type == cpp_PartitioningSpec.cpp_Type.NONE:
        return None
    elif spec.type == cpp_PartitioningSpec.cpp_Type.INHERIT:
        return "inherit"
    elif spec.type == cpp_PartitioningSpec.cpp_Type.HASH:
        return HashScheme.from_cpp(deref(spec.hash))
    elif spec.type == cpp_PartitioningSpec.cpp_Type.ORDER:
        return OrderScheme.from_cpp(deref(spec.order))
    else:
        raise ValueError("Unknown PartitioningSpec.Type")


cdef class Partitioning:
    """
    Hierarchical partitioning metadata for a data stream.

    Parameters
    ----------
    inter_rank
        Distribution across ranks. Can be a HashScheme, OrderScheme, None,
        or 'inherit'.
    local
        Distribution within a rank. Can be a HashScheme, OrderScheme, None,
        or 'inherit'.
    """

    def __init__(self, inter_rank=None, local=None):
        self._handle = make_unique[cpp_Partitioning]()
        deref(self._handle).inter_rank = _to_spec(inter_rank)
        deref(self._handle).local = _to_spec(local)

    def __dealloc__(self):
        with nogil:
            self._handle.reset()

    @staticmethod
    cdef Partitioning from_handle(unique_ptr[cpp_Partitioning] handle):
        cdef Partitioning ret = Partitioning.__new__(Partitioning)
        ret._handle = move(handle)
        return ret

    @property
    def inter_rank(self):
        return _from_spec(deref(self._handle).inter_rank)

    @property
    def local(self):
        return _from_spec(deref(self._handle).local)

    def __eq__(self, other):
        if not isinstance(other, Partitioning):
            return NotImplemented
        return deref(self._handle) == deref((<Partitioning>other)._handle)

    def __repr__(self):
        return f"Partitioning(inter_rank={self.inter_rank!r}, local={self.local!r})"


cdef class ChannelMetadata:
    """
    Channel-level metadata describing a data stream.

    Parameters
    ----------
    local_count
        Estimated number of chunks for this rank.
    partitioning
        How the data is partitioned (default: no partitioning).
    duplicated
        Whether data is duplicated on all workers (default: False).
    """

    def __init__(
        self,
        int local_count,
        *,
        partitioning: Partitioning | None = None,
        bint duplicated = False,
    ):
        if local_count < 0:
            raise ValueError(f"local_count must be non-negative, got {local_count}")

        cdef cpp_Partitioning part
        if partitioning is not None:
            part = deref((<Partitioning>partitioning)._handle)

        self._handle = make_unique[cpp_ChannelMetadata](
            local_count, part, duplicated
        )

    def __dealloc__(self):
        with nogil:
            self._handle.reset()

    @staticmethod
    cdef ChannelMetadata from_handle(unique_ptr[cpp_ChannelMetadata] handle):
        cdef ChannelMetadata ret = ChannelMetadata.__new__(ChannelMetadata)
        ret._handle = move(handle)
        return ret

    @staticmethod
    def from_message(Message message not None):
        """Construct by consuming a Message (message becomes empty)."""
        return ChannelMetadata.from_handle(
            cpp_channel_metadata_from_message(move(message._handle))
        )

    def into_message(self, uint64_t sequence_number, Message message not None):
        """Move this ChannelMetadata into a Message (invoked by Message.__init__)."""
        if not message.empty():
            raise ValueError("cannot move into a non-empty message")
        message._handle = cpp_to_message_channel_metadata(
            sequence_number, move(self.release_handle())
        )

    cdef const cpp_ChannelMetadata* handle_ptr(self) except NULL:
        """Return pointer to underlying handle, raising if released."""
        if not self._handle:
            raise ValueError("ChannelMetadata is uninitialized, has it been released?")
        return self._handle.get()

    @property
    def local_count(self) -> int:
        return self.handle_ptr().local_count

    @property
    def partitioning(self) -> Partitioning:
        cdef Partitioning ret = Partitioning.__new__(Partitioning)
        ret._handle = make_unique[cpp_Partitioning](self.handle_ptr().partitioning)
        return ret

    @property
    def duplicated(self) -> bool:
        return self.handle_ptr().duplicated

    def __eq__(self, other):
        if not isinstance(other, ChannelMetadata):
            return NotImplemented
        return deref(self.handle_ptr()) == deref((<ChannelMetadata>other).handle_ptr())

    def __repr__(self):
        return (
            f"ChannelMetadata(local_count={self.local_count}, "
            f"partitioning={self.partitioning!r}, "
            f"duplicated={self.duplicated})"
        )

    cdef unique_ptr[cpp_ChannelMetadata] release_handle(self):
        if not self._handle:
            raise ValueError("is uninitialized, has it been released?")
        return move(self._handle)
