# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Channel metadata types for streaming pipelines."""

from cython.operator cimport dereference as deref
from libc.stdint cimport uint64_t
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector

from rapidsmpf.streaming.core.message cimport Message


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
    """Hash partitioning scheme: rows distributed by hash(columns) % modulus."""

    def __init__(self, tuple columns, int modulus):
        cdef vector[string] cols
        for c in columns:
            cols.push_back((<str>c).encode())
        self._scheme = cpp_HashScheme(cols, modulus)

    @staticmethod
    cdef HashScheme from_cpp(cpp_HashScheme scheme):
        cdef HashScheme ret = HashScheme.__new__(HashScheme)
        ret._scheme = move(scheme)
        return ret

    @property
    def columns(self) -> tuple:
        return tuple(c.decode() for c in self._scheme.columns)

    @property
    def modulus(self) -> int:
        return self._scheme.modulus

    def __eq__(self, other):
        if not isinstance(other, HashScheme):
            return NotImplemented
        return self._scheme == (<HashScheme>other)._scheme

    def __repr__(self):
        return f"HashScheme({self.columns!r}, {self.modulus})"


cdef cpp_PartitioningSpec _to_spec(obj) except *:
    """Convert Python object to PartitioningSpec."""
    if obj is None:
        return cpp_PartitioningSpec.none()
    elif obj == "aligned":
        return cpp_PartitioningSpec.aligned()
    elif isinstance(obj, HashScheme):
        return cpp_PartitioningSpec.from_hash((<HashScheme>obj)._scheme)
    else:
        raise TypeError(
            f"Expected HashScheme, None, or 'aligned', got {type(obj).__name__}"
        )


cdef object _from_spec(cpp_PartitioningSpec spec):
    """Convert PartitioningSpec to Python object."""
    if spec.type == cpp_SpecType.NONE:
        return None
    elif spec.type == cpp_SpecType.ALIGNED:
        return "aligned"
    elif spec.type == cpp_SpecType.HASH:
        return HashScheme.from_cpp(deref(spec.hash))
    else:
        raise ValueError("Unknown SpecType")


cdef class Partitioning:
    """Hierarchical partitioning metadata (inter_rank and local levels)."""

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
    """Channel-level metadata: counts, partitioning, and duplication status."""

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

    @property
    def local_count(self) -> int:
        return deref(self._handle).local_count

    @property
    def partitioning(self) -> Partitioning:
        cdef Partitioning ret = Partitioning.__new__(Partitioning)
        ret._handle = make_unique[cpp_Partitioning](deref(self._handle).partitioning)
        return ret

    @property
    def duplicated(self) -> bool:
        return deref(self._handle).duplicated

    def __eq__(self, other):
        if not isinstance(other, ChannelMetadata):
            return NotImplemented
        return deref(self._handle) == deref((<ChannelMetadata>other)._handle)

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
