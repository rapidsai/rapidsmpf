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

    Equality (``==``) matches the C++ definition: same keys and strict_boundaries,
    and boundary tables with the same shape if present; boundary *cell values*
    are not compared.

    Parameters
    ----------
    keys
        Sequence of ``OrderKey`` objects (one per sort column).
    boundaries
        Optional ``TableChunk`` of N-1 boundary rows for N partitions.
    strict_boundaries
        When true, every row in a chunk falls in a single partition's half-open key
        range (keys do not straddle chunk interiors). See the C++ ``OrderScheme`` docs.
        Default false.
    """

    def __init__(
        self,
        object keys,
        TableChunk boundaries = None,
        *,
        bint strict_boundaries = False,
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
            _make_order_scheme(move(cpp_keys), move(bd_ptr), strict_boundaries)
        )

    @staticmethod
    cdef OrderScheme from_cpp(cpp_OrderScheme scheme):
        cdef OrderScheme ret = OrderScheme.__new__(OrderScheme)
        ret._storage = move(scheme)
        return ret

    @property
    def keys(self) -> tuple:
        cdef int i
        cdef int n = <int>self._storage.keys.size()
        return tuple(OrderKey.from_cpp(self._storage.keys[i]) for i in range(n))

    @property
    def strict_boundaries(self) -> bool:
        """Same semantics as the C++ ``OrderScheme::strict_boundaries`` field."""
        return self._storage.strict_boundaries

    @property
    def num_boundaries(self):
        """Number of boundary rows, or ``None`` if no boundaries."""
        if self._storage.boundaries.get() == NULL:
            return None
        return self._storage.boundaries.get().shape().first

    def get_boundaries_table(self):
        """Return boundary rows as ``pylibcudf.Table``, or ``None``.

        Raises if ``boundaries`` is set but not device-resident.
        """
        if self._storage.boundaries.get() == NULL:
            return None
        if not self._storage.boundaries.get().is_available():
            raise RuntimeError("ORDER boundaries must be device-resident")
        cdef cpp_table_view view = self._storage.boundaries.get().table_view()
        cdef cudaStream_t stream = self._storage.boundaries.get().stream().value()
        return Table.from_table_view_of_arbitrary(
            view, owner=self, stream=Stream._from_cudaStream_t(stream)
        )

    def __eq__(self, other):
        if not isinstance(other, OrderScheme):
            return NotImplemented
        return self._storage == (<OrderScheme>other)._storage

    def __repr__(self):
        has_b = self._storage.boundaries.get() != NULL
        return (
            f"OrderScheme({self.keys!r}, has_boundaries={has_b}, "
            f"strict_boundaries={self.strict_boundaries})"
        )


cdef void _apply_spec(cpp_PartitioningSpec& spec, obj) except *:
    """Set *spec* in-place from a Python value."""
    if obj is None:
        spec = cpp_PartitioningSpec.none()
    elif obj == "inherit":
        spec = cpp_PartitioningSpec.inherit()
    elif isinstance(obj, HashScheme):
        spec = cpp_PartitioningSpec.from_hash((<HashScheme>obj)._scheme)
    elif isinstance(obj, OrderScheme):
        spec = cpp_PartitioningSpec.from_order((<OrderScheme>obj)._storage)
    else:
        raise TypeError(
            f"Expected HashScheme, OrderScheme, None, or 'inherit', "
            f"got {type(obj).__name__}"
        )


cdef object _from_spec(cpp_PartitioningSpec& spec):
    """Convert PartitioningSpec (by reference) to a Python object."""
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
        _apply_spec(self._data.inter_rank, inter_rank)
        _apply_spec(self._data.local, local)

    @staticmethod
    cdef Partitioning from_cpp(cpp_Partitioning data):
        cdef Partitioning ret = Partitioning.__new__(Partitioning)
        ret._data = move(data)
        return ret

    @property
    def inter_rank(self):
        return _from_spec(self._data.inter_rank)

    @property
    def local(self):
        return _from_spec(self._data.local)

    def __eq__(self, other):
        if not isinstance(other, Partitioning):
            return NotImplemented
        return self._data == (<Partitioning>other)._data

    def __repr__(self):
        return f"Partitioning(inter_rank={self.inter_rank!r}, local={self.local!r})"


cdef extern from * nogil:
    """
    #include <memory>
    #include <rapidsmpf/streaming/cudf/channel_metadata.hpp>

    static rapidsmpf::streaming::OrderScheme _make_order_scheme(
        std::vector<rapidsmpf::streaming::OrderKey> keys,
        std::unique_ptr<rapidsmpf::streaming::TableChunk> boundaries,
        bool strict_boundaries
    ) {
        return rapidsmpf::streaming::OrderScheme{
            std::move(keys), std::move(boundaries), strict_boundaries
        };
    }

    static std::unique_ptr<rapidsmpf::streaming::ChannelMetadata>
    _channel_metadata_from_message(rapidsmpf::streaming::Message msg) {
        return std::make_unique<rapidsmpf::streaming::ChannelMetadata>(
            msg.release<rapidsmpf::streaming::ChannelMetadata>()
        );
    }
    """
    cpp_OrderScheme _make_order_scheme(
        vector[cpp_OrderKey], unique_ptr[cpp_TableChunk], bool_t
    ) except +
    unique_ptr[cpp_ChannelMetadata] _channel_metadata_from_message(
        cpp_Message
    ) except +


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

        if partitioning is not None:
            self._handle = make_unique[cpp_ChannelMetadata](
                local_count, (<Partitioning>partitioning)._data, duplicated
            )
        else:
            self._handle = make_unique[cpp_ChannelMetadata](
                local_count, cpp_Partitioning(), duplicated
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
            _channel_metadata_from_message(move(message._handle))
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
        return Partitioning.from_cpp(self._handle.get().partitioning)

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
