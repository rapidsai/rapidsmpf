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

import pylibcudf.types as plc_types


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


cdef class OrderKey:
    """A single sort key: column index, sort direction, and null placement."""

    def __init__(
        self,
        int column_index,
        cpp_order order,
        cpp_null_order null_order,
    ):
        self._key.column_index = column_index
        self._key.order = order
        self._key.null_order = null_order

    @staticmethod
    cdef OrderKey from_cpp(cpp_OrderKey key):
        cdef OrderKey ret = OrderKey.__new__(OrderKey)
        ret._key = key
        return ret

    @property
    def column_index(self) -> int:
        return self._key.column_index

    @property
    def order(self):
        return plc_types.Order(<int>self._key.order)

    @property
    def null_order(self):
        return plc_types.NullOrder(<int>self._key.null_order)

    def __eq__(self, other):
        if not isinstance(other, OrderKey):
            return NotImplemented
        return self._key == (<OrderKey>other)._key

    def __repr__(self):
        return f"OrderKey({self.column_index}, {self.order!r}, {self.null_order!r})"


cdef class OrderScheme:
    """Order-based partitioning scheme for sorted/range-partitioned data.

    Data is partitioned by value ranges based on predetermined boundaries.
    For N partitions, there are N-1 boundary rows.

    Equality (``==``) matches the C++ definition: same keys and strict_boundary,
    and boundary tables with the same shape if present; boundary *cell values*
    are not compared.

    Parameters
    ----------
    keys
        Sequence of ``OrderKey`` objects (one per sort column).
    boundaries
        Optional ``TableChunk`` of N-1 boundary rows for N partitions.
    strict_boundary
        When true, every row in a chunk falls in a single partition's half-open key
        range (keys do not straddle chunk interiors). See the C++ ``OrderScheme`` docs.
        Default false.

    Notes
    -----
    Passing this object to ``Partitioning(...)`` transfers ownership of any
    ``boundaries`` into the partitioning.  The source retains its keys but
    ``get_boundaries_table()`` will return ``None`` afterward.

    Metadata paths keep boundary ``TableChunk`` objects device-resident; if
    boundaries are packed or spilled, ``get_boundaries_table()`` raises (there is no
    unspill hook).
    """

    def __cinit__(self):
        self._ptr = &self._storage
        self._owner = None

    def __init__(
        self,
        object keys,
        TableChunk boundaries = None,
        *,
        bint strict_boundary = False,
    ):
        cdef vector[cpp_OrderKey] cpp_keys
        cdef unique_ptr[cpp_TableChunk] bd_ptr
        for key in keys:
            if not isinstance(key, OrderKey):
                raise TypeError(
                    f"keys must contain OrderKey objects, got {type(key).__name__}"
                )
            cpp_keys.push_back((<OrderKey>key)._key)
        if cpp_keys.empty():
            raise ValueError("OrderScheme: keys must not be empty")
        if boundaries is not None:
            bd_ptr = move(boundaries.release_handle())
        self._storage = move(
            make_order_scheme(move(cpp_keys), move(bd_ptr), strict_boundary)
        )
        self._ptr = &self._storage
        self._owner = None

    @staticmethod
    cdef OrderScheme from_cpp(cpp_OrderScheme scheme):
        cdef OrderScheme ret = OrderScheme.__new__(OrderScheme)
        ret._storage = move(scheme)
        ret._ptr = &ret._storage
        ret._owner = None
        return ret

    @staticmethod
    cdef OrderScheme view_of(cpp_OrderScheme* ptr, object owner):
        """Non-owning view of ``ptr``; ``owner`` keeps the backing C++ storage alive."""
        cdef OrderScheme ret = OrderScheme.__new__(OrderScheme)
        ret._ptr = ptr
        ret._owner = owner
        return ret

    cdef cpp_OrderScheme* _get(self):
        return self._ptr

    @property
    def keys(self) -> tuple:
        cdef int i
        cdef int n = <int>self._get().keys.size()
        return tuple(OrderKey.from_cpp(self._get().keys[i]) for i in range(n))

    @property
    def strict_boundary(self) -> bool:
        """Same semantics as the C++ ``OrderScheme::strict_boundary`` field."""
        return self._get().strict_boundary

    @property
    def num_boundaries(self):
        """Number of boundary rows, or ``None`` if no boundaries (shape-based)."""
        if self._get().boundaries == NULL:
            return None
        return order_scheme_boundary_row_count(self._get())

    def get_boundaries_table(self):
        """Return boundary rows as ``pylibcudf.Table``, or ``None``.

        Raises if ``boundaries`` is set but not device-resident (metadata does not
        unspill boundary tables).
        """
        if self._get().boundaries == NULL:
            return None
        cdef cpp_table_view view = order_scheme_boundaries_table_view(self._get())
        cdef cudaStream_t stream = order_scheme_boundaries_cuda_stream(self._get())
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
            f"OrderScheme({self.keys!r}, has_boundaries={has_b}, "
            f"strict_boundary={self.strict_boundary})"
        )


cdef void _apply_spec(cpp_PartitioningSpec& spec, obj) except *:
    """Set *spec* in-place from a Python object (avoids copying move-only types)."""
    if obj is None:
        partitioning_spec_set_none(spec)
    elif obj == "inherit":
        partitioning_spec_set_inherit(spec)
    elif isinstance(obj, HashScheme):
        partitioning_spec_set_hash(spec, (<HashScheme>obj)._scheme)
    elif isinstance(obj, OrderScheme):
        # Copies keys; moves boundaries unique_ptr (source loses boundary ownership).
        # Use _get() for view-mode OrderSchemes (e.g. metadata.partitioning.inter_rank)
        # so we target the live cpp_OrderScheme, not the wrapper's empty owning slot.
        partitioning_spec_set_order(spec, deref((<OrderScheme>obj)._get()))
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
        return OrderScheme.view_of(partitioning_spec_order_scheme_ptr(spec), owner)
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
            self._handle = make_channel_metadata(
                local_count, deref((<Partitioning>partitioning)._get()), duplicated
            )
        else:
            self._handle = make_channel_metadata(local_count, empty_part, duplicated)

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
            channel_metadata_from_message(move(message._handle))
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
