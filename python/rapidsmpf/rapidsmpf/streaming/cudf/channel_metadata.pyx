# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Channel metadata types for streaming pipelines."""

# cudaStream_t: boundaries TableChunk exposes libcudf's stream handle;
# wrap via Stream._from_cudaStream_t (see streaming/core/context.pyx).
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

from pylibcudf.types import NullOrder as PyNullOrder
from pylibcudf.types import Order as PyOrder


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

    def __init__(self, object column_indices, int modulus):
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


cdef cpp_order _object_to_cpp_order(object o) except *:
    """Map pylibcudf.types.Order (int-valued enum) to libcudf order."""
    cdef int v = int(o)
    if v == <int>cpp_order.ASCENDING:
        return cpp_order.ASCENDING
    if v == <int>cpp_order.DESCENDING:
        return cpp_order.DESCENDING
    raise ValueError(
        f"Invalid order: {o!r}; use pylibcudf.types.Order.ASCENDING or Order.DESCENDING"
    )


cdef cpp_null_order _object_to_cpp_null_order(object o) except *:
    """Map pylibcudf.types.NullOrder (int-valued enum) to libcudf null_order."""
    cdef int v = int(o)
    if v == <int>cpp_null_order.BEFORE:
        return cpp_null_order.BEFORE
    if v == <int>cpp_null_order.AFTER:
        return cpp_null_order.AFTER
    raise ValueError(
        f"Invalid null order: {o!r}; use pylibcudf.types.NullOrder.BEFORE or "
        f"NullOrder.AFTER (libcudf names; BEFORE/AFTER null placement in sort)"
    )


cdef class OrderScheme:
    """Order-based partitioning scheme for sorted/range-partitioned data.

    Data is partitioned by value ranges based on predetermined boundaries.
    For N partitions, there are N-1 boundary rows.

    Equality (``==``) matches the C++ definition: same indices, orders,
    null_orders, strict_boundary, and boundary tables with the same shape if
    present; boundary *cell values* are not compared.

    Parameters
    ----------
    column_indices, orders, null_orders
        Per sort key: use ``pylibcudf.types.Order`` and ``pylibcudf.types.NullOrder``
        (libcudf ``BEFORE``/``AFTER`` naming for null placement).
    strict_boundary
        If True, sort keys do not span partition interiors so merge-style
        algorithms may skip halo exchange. Default False.
    """

    def __init__(
        self,
        object column_indices,
        object orders,
        object null_orders,
        TableChunk boundaries = None,
        *,
        bint strict_boundary = False,
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
            ords.push_back(_object_to_cpp_order(o))
        for n in null_orders:
            nulls.push_back(_object_to_cpp_null_order(n))

        self._scheme.column_indices = move(cols)
        self._scheme.orders = move(ords)
        self._scheme.null_orders = move(nulls)

        if boundaries is not None:
            # Move the TableChunk's handle into a shared_ptr (consumes the TableChunk)
            self._scheme.boundaries = unique_to_shared(boundaries.release_handle())

        self._scheme.strict_boundary = strict_boundary

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
        cdef int i
        cdef int n = <int>self._scheme.orders.size()
        cdef cpp_order co
        out = []
        for i in range(n):
            co = self._scheme.orders[i]
            out.append(PyOrder(<int>co))
        return tuple(out)

    @property
    def null_orders(self) -> tuple:
        cdef int i
        cdef int n = <int>self._scheme.null_orders.size()
        cdef cpp_null_order no
        out = []
        for i in range(n):
            no = self._scheme.null_orders[i]
            out.append(PyNullOrder(<int>no))
        return tuple(out)

    @property
    def has_boundaries(self) -> bool:
        """True if a boundaries table is attached.

        Cheap; does not build a ``Table``.
        """
        return has_chunk(self._scheme.boundaries)

    @property
    def strict_boundary(self) -> bool:
        """True if sort keys do not span partition interiors (strict ranges)."""
        return self._scheme.strict_boundary

    def get_boundaries_table(self):
        """Return boundary rows as ``pylibcudf.Table``, or ``None``.

        Prefer :attr:`has_boundaries` for a boolean. This builds a ``Table`` view
        tied to this object's lifetime; keep use to tests and consumers that must
        inspect boundary values (e.g. cudf-polars round-trip checks).
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
            f"{self.null_orders!r}, has_boundaries={self.has_boundaries}, "
            f"strict_boundary={self.strict_boundary})"
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
