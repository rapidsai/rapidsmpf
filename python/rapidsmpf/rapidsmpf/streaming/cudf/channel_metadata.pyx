# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Channel metadata types for streaming pipelines."""

from cython.operator cimport dereference as deref
from libc.stdint cimport int64_t, uint64_t
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.optional cimport optional
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector

from rapidsmpf.streaming.core.message cimport Message


cdef class HashScheme:
    """
    Hash partitioning scheme.

    Rows are distributed by ``hash(columns) % modulus``.

    Parameters
    ----------
    columns : tuple[str, ...]
        Column names to hash on.
    modulus : int
        Hash modulus (number of partitions).

    Examples
    --------
    >>> scheme = HashScheme(("key",), 16)
    >>> scheme.columns
    ('key',)
    >>> scheme.modulus
    16
    """

    def __init__(self, tuple columns, int modulus):
        cdef vector[string] cols
        for c in columns:
            cols.push_back((<str>c).encode())
        self._scheme = cpp_HashScheme(cols, modulus)

    @staticmethod
    cdef HashScheme from_cpp(cpp_HashScheme scheme):
        """Construct a HashScheme from an existing C++ HashScheme."""
        cdef HashScheme ret = HashScheme.__new__(HashScheme)
        ret._scheme = move(scheme)
        return ret

    @property
    def columns(self) -> tuple:
        """Column names to hash on."""
        return tuple(c.decode() for c in self._scheme.columns)

    @property
    def modulus(self) -> int:
        """Hash modulus (number of partitions)."""
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
    if spec.is_none():
        return None
    elif spec.is_aligned():
        return "aligned"
    elif spec.is_hash():
        return HashScheme.from_cpp(deref(spec.hash))
    else:
        raise ValueError("Unknown SpecType")


cdef class Partitioning:
    """
    Hierarchical partitioning metadata for a data stream.

    Describes how data flowing through a channel is partitioned at multiple
    levels of the system hierarchy:

    - ``inter_rank``: Distribution across ranks (global partitioning).
    - ``local``: Distribution within a rank (local chunk assignment).

    Parameters
    ----------
    inter_rank : HashScheme | None | "aligned"
        Inter-rank partitioning specification:

        - ``HashScheme``: Explicit hash partitioning was applied at this level.
        - ``None``: No partitioning at this level.
        - ``"aligned"``: Partitioning is inherited from the parent level.
    local : HashScheme | None | "aligned"
        Local partitioning specification (same options as ``inter_rank``).

    Examples
    --------
    Direct global shuffle to N_g partitions:

    >>> p = Partitioning(HashScheme(("key",), N_g), "aligned")

    Two-stage shuffle (global by nranks, then local to N_l):

    >>> p = Partitioning(HashScheme(("key",), nranks), HashScheme(("key",), N_l))

    After local repartition (lose local alignment):

    >>> p = Partitioning(HashScheme(("key",), N_g), None)
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
        """Construct a Partitioning from an existing C++ handle."""
        cdef Partitioning ret = Partitioning.__new__(Partitioning)
        ret._handle = move(handle)
        return ret

    @staticmethod
    def from_message(Message message not None):
        """
        Construct a Partitioning by consuming a Message.

        Parameters
        ----------
        message
            Message containing a Partitioning. The message is released
            and is empty after this call.

        Returns
        -------
        Partitioning
            A new Partitioning extracted from the given message.
        """
        return Partitioning.from_handle(
            make_unique[cpp_Partitioning](
                message._handle.release[cpp_Partitioning]()
            )
        )

    def into_message(self, uint64_t sequence_number, Message message not None):
        """
        Move this Partitioning into a Message.

        This method is not typically called directly. Instead, it is invoked by
        ``Message.__init__()`` when creating a new Message with this Partitioning
        as its payload.

        Parameters
        ----------
        sequence_number
            Ordering identifier for the message.
        message
            Message object that will take ownership of this Partitioning.

        Raises
        ------
        ValueError
            If the provided message is not empty.

        Warnings
        --------
        The Partitioning is released and must not be used after this call.
        """
        if not message.empty():
            raise ValueError("cannot move into a non-empty message")
        message._handle = cpp_to_message_partitioning(
            sequence_number, move(self.release_handle())
        )

    @property
    def inter_rank(self):
        """
        Inter-rank partitioning specification.

        Returns
        -------
        HashScheme | None | str
            - ``HashScheme``: Explicit hash partitioning.
            - ``None``: No partitioning.
            - ``"aligned"``: Aligned with parent level.
        """
        return _from_spec(deref(self._handle).inter_rank)

    @property
    def local(self):
        """
        Local partitioning specification.

        Returns
        -------
        HashScheme | None | str
            - ``HashScheme``: Explicit hash partitioning.
            - ``None``: No partitioning.
            - ``"aligned"``: Aligned with parent level.
        """
        return _from_spec(deref(self._handle).local)

    def __eq__(self, other):
        if not isinstance(other, Partitioning):
            return NotImplemented
        return deref(self._handle) == deref((<Partitioning>other)._handle)

    def __repr__(self):
        return f"Partitioning(inter_rank={self.inter_rank!r}, local={self.local!r})"

    cdef const cpp_Partitioning* handle_ptr(self):
        """
        Return a pointer to the underlying C++ Partitioning.

        Returns
        -------
        Raw pointer to the underlying C++ object.

        Raises
        ------
        ValueError
            If the Partitioning is uninitialized.
        """
        if not self._handle:
            raise ValueError("is uninitialized, has it been released?")
        return self._handle.get()

    cdef unique_ptr[cpp_Partitioning] release_handle(self):
        """
        Release ownership of the underlying C++ Partitioning.

        After this call, the current object is in a moved-from state and
        must not be accessed.

        Returns
        -------
        Unique pointer to the underlying C++ object.

        Raises
        ------
        ValueError
            If the Partitioning is uninitialized.
        """
        if not self._handle:
            raise ValueError("is uninitialized, has it been released?")
        return move(self._handle)


cdef class ChannelMetadata:
    """
    Channel-level metadata describing the data stream.

    Contains information about chunk counts, partitioning, and duplication
    status for the data flowing through a channel.

    Parameters
    ----------
    local_count : int
        Local chunk-count estimate for this rank (must be >= 0).
    global_count : int | None
        Global chunk-count estimate across all ranks.
    partitioning : Partitioning | None
        How the data is partitioned.
    duplicated : bool
        Whether data is duplicated (identical) on all workers.

    Examples
    --------
    >>> m = ChannelMetadata(
    ...     local_count=4,
    ...     global_count=16,
    ...     partitioning=Partitioning(HashScheme(("key",), 16), "aligned"),
    ...     duplicated=False,
    ... )
    """

    def __init__(
        self,
        int local_count,
        *,
        global_count: int | None = None,
        partitioning: Partitioning | None = None,
        bint duplicated = False,
    ):
        if local_count < 0:
            raise ValueError(f"local_count must be non-negative, got {local_count}")
        if global_count is not None and global_count < 0:
            raise ValueError(f"global_count must be non-negative, got {global_count}")

        cdef optional[int64_t] gc
        if global_count is not None:
            gc = <int64_t>global_count

        cdef cpp_Partitioning part
        if partitioning is not None:
            part = deref((<Partitioning>partitioning)._handle)

        self._handle = make_unique[cpp_ChannelMetadata](
            local_count, gc, part, duplicated
        )

    def __dealloc__(self):
        with nogil:
            self._handle.reset()

    @staticmethod
    cdef ChannelMetadata from_handle(unique_ptr[cpp_ChannelMetadata] handle):
        """Construct a ChannelMetadata from an existing C++ handle."""
        cdef ChannelMetadata ret = ChannelMetadata.__new__(ChannelMetadata)
        ret._handle = move(handle)
        return ret

    @staticmethod
    def from_message(Message message not None):
        """
        Construct a ChannelMetadata by consuming a Message.

        Parameters
        ----------
        message
            Message containing a ChannelMetadata. The message is released
            and is empty after this call.

        Returns
        -------
        ChannelMetadata
            A new ChannelMetadata extracted from the given message.
        """
        return ChannelMetadata.from_handle(
            make_unique[cpp_ChannelMetadata](
                message._handle.release[cpp_ChannelMetadata]()
            )
        )

    def into_message(self, uint64_t sequence_number, Message message not None):
        """
        Move this ChannelMetadata into a Message.

        This method is not typically called directly. Instead, it is invoked by
        ``Message.__init__()`` when creating a new Message with this ChannelMetadata
        as its payload.

        Parameters
        ----------
        sequence_number
            Ordering identifier for the message.
        message
            Message object that will take ownership of this ChannelMetadata.

        Raises
        ------
        ValueError
            If the provided message is not empty.

        Warnings
        --------
        The ChannelMetadata is released and must not be used after this call.
        """
        if not message.empty():
            raise ValueError("cannot move into a non-empty message")
        message._handle = cpp_to_message_channel_metadata(
            sequence_number, move(self.release_handle())
        )

    @property
    def local_count(self) -> int:
        """Local chunk-count estimate for this rank."""
        return deref(self._handle).local_count

    @property
    def global_count(self) -> int | None:
        """Global chunk-count estimate across all ranks."""
        cdef optional[int64_t] gc = deref(self._handle).global_count
        if gc.has_value():
            return gc.value()
        return None

    @property
    def partitioning(self) -> Partitioning:
        """How the data is partitioned."""
        cdef Partitioning ret = Partitioning.__new__(Partitioning)
        ret._handle = make_unique[cpp_Partitioning](deref(self._handle).partitioning)
        return ret

    @property
    def duplicated(self) -> bool:
        """Whether data is duplicated (identical) on all workers."""
        return deref(self._handle).duplicated

    def __eq__(self, other):
        if not isinstance(other, ChannelMetadata):
            return NotImplemented
        return deref(self._handle) == deref((<ChannelMetadata>other)._handle)

    def __repr__(self):
        return (
            f"ChannelMetadata(local_count={self.local_count}, "
            f"global_count={self.global_count}, "
            f"partitioning={self.partitioning!r}, "
            f"duplicated={self.duplicated})"
        )

    cdef const cpp_ChannelMetadata* handle_ptr(self):
        """
        Return a pointer to the underlying C++ ChannelMetadata.

        Returns
        -------
        Raw pointer to the underlying C++ object.

        Raises
        ------
        ValueError
            If the ChannelMetadata is uninitialized.
        """
        if not self._handle:
            raise ValueError("is uninitialized, has it been released?")
        return self._handle.get()

    cdef unique_ptr[cpp_ChannelMetadata] release_handle(self):
        """
        Release ownership of the underlying C++ ChannelMetadata.

        After this call, the current object is in a moved-from state and
        must not be accessed.

        Returns
        -------
        Unique pointer to the underlying C++ object.

        Raises
        ------
        ValueError
            If the ChannelMetadata is uninitialized.
        """
        if not self._handle:
            raise ValueError("is uninitialized, has it been released?")
        return move(self._handle)
