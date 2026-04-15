# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Channel metadata types for streaming pipelines."""

from cuda.bindings.cyruntime cimport cudaStream_t
from cython.operator cimport dereference as deref
from libc.stdint cimport int32_t, uint64_t
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector
from pylibcudf.libcudf.table.table_view cimport table_view as cpp_table_view
from pylibcudf.libcudf.types cimport null_order as cpp_null_order
from pylibcudf.libcudf.types cimport order as cpp_order
from pylibcudf.table cimport Table
from rmm.pylibrmm.stream cimport Stream

from rapidsmpf.streaming.core.message cimport Message
from rapidsmpf.streaming.cudf.table_chunk cimport TableChunk

from pylibcudf.types import NullOrder as PyNullOrder
from pylibcudf.types import Order as PyOrder


cdef extern from * nogil:
    """
    #include <rapidsmpf/streaming/cudf/table_chunk.hpp>
    #include <rapidsmpf/streaming/cudf/channel_metadata.hpp>
    #include <memory>
    namespace {

    // In-place spec setters — use move-assignment to avoid copy of move-only types.
    void _spec_set_none(rapidsmpf::streaming::PartitioningSpec& s) {
        s = rapidsmpf::streaming::PartitioningSpec::none();
    }
    void _spec_set_inherit(rapidsmpf::streaming::PartitioningSpec& s) {
        s = rapidsmpf::streaming::PartitioningSpec::inherit();
    }
    void _spec_set_hash(rapidsmpf::streaming::PartitioningSpec& s,
                        rapidsmpf::streaming::HashScheme h) {
        s = rapidsmpf::streaming::PartitioningSpec::from_hash(std::move(h));
    }
    // Copies vectors, moves boundaries (source keeps vectors, loses ownership).
    void _spec_set_order(rapidsmpf::streaming::PartitioningSpec& s,
                         rapidsmpf::streaming::OrderScheme& src) {
        rapidsmpf::streaming::OrderScheme o{
            .column_indices = src.column_indices,
            .orders         = src.orders,
            .null_orders    = src.null_orders,
            .boundaries     = std::move(src.boundaries),
            .strict_boundary = src.strict_boundary,
        };
        s = rapidsmpf::streaming::PartitioningSpec::from_order(std::move(o));
    }

    // Return a table_view from the raw OrderScheme pointer.
    cudf::table_view _order_scheme_boundaries_view(
        const rapidsmpf::streaming::OrderScheme* p
    ) {
        return p->boundaries->table_view();
    }

    // Return the cuda stream from the raw OrderScheme pointer.
    cudaStream_t _order_scheme_boundaries_stream(
        const rapidsmpf::streaming::OrderScheme* p
    ) {
        return p->boundaries->stream().value();
    }

    // Retrieve a raw pointer to the OrderScheme inside an ORDER PartitioningSpec.
    rapidsmpf::streaming::OrderScheme* _spec_order_ptr(
        rapidsmpf::streaming::PartitioningSpec& spec
    ) {
        return &spec.order.value();
    }

    std::unique_ptr<rapidsmpf::streaming::ChannelMetadata>
    cpp_channel_metadata_from_message(rapidsmpf::streaming::Message msg) {
        return std::make_unique<rapidsmpf::streaming::ChannelMetadata>(
            msg.release<rapidsmpf::streaming::ChannelMetadata>()
        );
    }

    // Construct ChannelMetadata by shallow-cloning a Partitioning:
    // copies hash specs and vectors; moves any ORDER boundaries (source loses them).
    std::unique_ptr<rapidsmpf::streaming::ChannelMetadata>
    _make_channel_metadata(
        uint64_t local_count,
        rapidsmpf::streaming::Partitioning& p,
        bool duplicated
    ) {
        auto clone_spec = [](rapidsmpf::streaming::PartitioningSpec& spec)
            -> rapidsmpf::streaming::PartitioningSpec
        {
            using T = rapidsmpf::streaming::PartitioningSpec::Type;
            switch (spec.type) {
                case T::NONE:
                    return rapidsmpf::streaming::PartitioningSpec::none();
                case T::INHERIT:
                    return rapidsmpf::streaming::PartitioningSpec::inherit();
                case T::HASH:
                    return rapidsmpf::streaming::PartitioningSpec::from_hash(
                        *spec.hash
                    );
                case T::ORDER: {
                    rapidsmpf::streaming::OrderScheme o{
                        .column_indices  = spec.order->column_indices,
                        .orders          = spec.order->orders,
                        .null_orders     = spec.order->null_orders,
                        .boundaries      = std::move(spec.order->boundaries),
                        .strict_boundary = spec.order->strict_boundary,
                    };
                    return rapidsmpf::streaming::PartitioningSpec::from_order(
                        std::move(o)
                    );
                }
            }
            return rapidsmpf::streaming::PartitioningSpec::none();
        };
        rapidsmpf::streaming::Partitioning part{
            clone_spec(p.inter_rank), clone_spec(p.local)
        };
        return std::make_unique<rapidsmpf::streaming::ChannelMetadata>(
            local_count, std::move(part), duplicated
        );
    }
    }
    """
    void _spec_set_none(cpp_PartitioningSpec&) noexcept
    void _spec_set_inherit(cpp_PartitioningSpec&) noexcept
    void _spec_set_hash(cpp_PartitioningSpec&, cpp_HashScheme) noexcept
    void _spec_set_order(cpp_PartitioningSpec&, cpp_OrderScheme&) except +
    cpp_table_view _order_scheme_boundaries_view(cpp_OrderScheme*) except +
    cudaStream_t _order_scheme_boundaries_stream(cpp_OrderScheme*) noexcept
    cpp_OrderScheme* _spec_order_ptr(cpp_PartitioningSpec&) noexcept
    unique_ptr[cpp_ChannelMetadata] cpp_channel_metadata_from_message(
        cpp_Message
    ) except +
    unique_ptr[cpp_ChannelMetadata] _make_channel_metadata(
        uint64_t, cpp_Partitioning&, bint
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

    Notes
    -----
    Passing this object to ``Partitioning(...)`` transfers ownership of any
    ``boundaries`` data into the partitioning.  The source object retains its
    column/order vectors, but ``get_boundaries_table()`` will return ``None``
    afterward.
    """

    def __cinit__(self):
        self._view = NULL
        self._owner = None

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
            self._scheme.boundaries = move(boundaries.release_handle())

        self._scheme.strict_boundary = strict_boundary

    @staticmethod
    cdef OrderScheme from_cpp(cpp_OrderScheme scheme):
        cdef OrderScheme ret = OrderScheme.__new__(OrderScheme)
        ret._scheme = move(scheme)
        return ret

    @staticmethod
    cdef OrderScheme view_of(cpp_OrderScheme* ptr, object owner):
        """Return a non-owning OrderScheme backed by *ptr; owner kept alive."""
        cdef OrderScheme ret = OrderScheme.__new__(OrderScheme)
        ret._view = ptr
        ret._owner = owner
        return ret

    cdef cpp_OrderScheme* _get(self):
        return self._view if self._view != NULL else &self._scheme

    @property
    def column_indices(self) -> tuple:
        return tuple(self._get().column_indices)

    @property
    def orders(self) -> tuple:
        cdef int i
        cdef int n = <int>self._get().orders.size()
        cdef cpp_order co
        out = []
        for i in range(n):
            co = self._get().orders[i]
            out.append(PyOrder(<int>co))
        return tuple(out)

    @property
    def null_orders(self) -> tuple:
        cdef int i
        cdef int n = <int>self._get().null_orders.size()
        cdef cpp_null_order no
        out = []
        for i in range(n):
            no = self._get().null_orders[i]
            out.append(PyNullOrder(<int>no))
        return tuple(out)

    @property
    def strict_boundary(self) -> bool:
        """True if sort keys do not span partition interiors (strict ranges)."""
        return self._get().strict_boundary

    def get_boundaries_table(self):
        """Return boundary rows as ``pylibcudf.Table``, or ``None``.

        Builds a ``Table`` view tied to this object's lifetime; keep use to
        tests and consumers that must inspect boundary values (e.g. cudf-polars
        round-trip checks).
        """
        if self._get().boundaries == NULL:
            return None
        cdef cpp_table_view view = _order_scheme_boundaries_view(self._get())
        cdef cudaStream_t stream = _order_scheme_boundaries_stream(self._get())
        # owner=self keeps the OrderScheme (and its keepalive chain) alive
        return Table.from_table_view_of_arbitrary(
            view, owner=self, stream=Stream._from_cudaStream_t(stream)
        )

    def __eq__(self, other):
        if not isinstance(other, OrderScheme):
            return NotImplemented
        return deref(self._get()) == deref((<OrderScheme>other)._get())

    def __repr__(self):
        has_b = self._get().boundaries != NULL
        return (
            f"OrderScheme({self.column_indices!r}, {self.orders!r}, "
            f"{self.null_orders!r}, has_boundaries={has_b}, "
            f"strict_boundary={self.strict_boundary})"
        )


cdef void _apply_spec(cpp_PartitioningSpec& spec, obj) except *:
    """Set *spec* in-place from a Python object (avoids copying move-only types)."""
    if obj is None:
        _spec_set_none(spec)
    elif obj == "inherit":
        _spec_set_inherit(spec)
    elif isinstance(obj, HashScheme):
        _spec_set_hash(spec, (<HashScheme>obj)._scheme)
    elif isinstance(obj, OrderScheme):
        # Copies vectors; moves boundaries unique_ptr (source loses boundary ownership).
        _spec_set_order(spec, (<OrderScheme>obj)._scheme)
    else:
        raise TypeError(
            f"Expected HashScheme, OrderScheme, None, or 'inherit', "
            f"got {type(obj).__name__}"
        )


cdef object _from_spec(cpp_PartitioningSpec& spec, object owner):
    """Convert PartitioningSpec (by reference) to Python object.

    For ORDER specs, returns a non-owning OrderScheme view backed by *owner*.
    """
    if spec.type == cpp_PartitioningSpec.cpp_Type.NONE:
        return None
    elif spec.type == cpp_PartitioningSpec.cpp_Type.INHERIT:
        return "inherit"
    elif spec.type == cpp_PartitioningSpec.cpp_Type.HASH:
        return HashScheme.from_cpp(deref(spec.hash))
    elif spec.type == cpp_PartitioningSpec.cpp_Type.ORDER:
        return OrderScheme.view_of(_spec_order_ptr(spec), owner)
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

    def __cinit__(self):
        self._ptr = NULL
        self._owner = None

    def __init__(self, inter_rank=None, local=None):
        self._handle = make_unique[cpp_Partitioning]()
        _apply_spec(deref(self._handle).inter_rank, inter_rank)
        _apply_spec(deref(self._handle).local, local)

    def __dealloc__(self):
        with nogil:
            self._handle.reset()

    @staticmethod
    cdef Partitioning from_handle(unique_ptr[cpp_Partitioning] handle):
        cdef Partitioning ret = Partitioning.__new__(Partitioning)
        ret._handle = move(handle)
        return ret

    @staticmethod
    cdef Partitioning view_of(cpp_Partitioning* ptr, object owner):
        """Return a non-owning Partitioning backed by *ptr; owner kept alive."""
        cdef Partitioning ret = Partitioning.__new__(Partitioning)
        ret._ptr = ptr
        ret._owner = owner
        return ret

    cdef cpp_Partitioning* _get(self):
        return self._ptr if self._ptr != NULL else self._handle.get()

    @property
    def inter_rank(self):
        return _from_spec(self._get().inter_rank, self)

    @property
    def local(self):
        return _from_spec(self._get().local, self)

    def __eq__(self, other):
        if not isinstance(other, Partitioning):
            return NotImplemented
        return deref(self._get()) == deref((<Partitioning>other)._get())

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

        cdef cpp_Partitioning empty_part
        if partitioning is not None:
            # Shallow-clone: copies hash specs and vectors, moves ORDER boundaries.
            # The source Partitioning retains its vectors but loses boundary ownership.
            self._handle = _make_channel_metadata(
                local_count, deref((<Partitioning>partitioning)._get()), duplicated
            )
        else:
            self._handle = _make_channel_metadata(local_count, empty_part, duplicated)

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
        if not self._handle:
            raise ValueError("ChannelMetadata is uninitialized, has it been released?")
        # Non-owning view backed by this object (keeps self alive).
        return Partitioning.view_of(&self._handle.get().partitioning, self)

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
