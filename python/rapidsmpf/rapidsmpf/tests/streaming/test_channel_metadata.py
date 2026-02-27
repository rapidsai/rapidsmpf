# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Tests for streaming metadata types (Partitioning and ChannelMetadata)."""

from __future__ import annotations

import pytest

import cudf
import rmm.pylibrmm.stream

from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.cudf import (
    ChannelMetadata,
    HashScheme,
    OrderScheme,
    Partitioning,
    TableChunk,
)
from rapidsmpf.utils.cudf import cudf_to_pylibcudf_table


def test_hash_scheme() -> None:
    """Test HashScheme construction, properties, equality, and repr."""
    h1 = HashScheme((0, 1), 16)
    assert h1.column_indices == (0, 1)
    assert h1.modulus == 16
    assert repr(h1) == "HashScheme((0, 1), 16)"

    # Equality
    assert h1 == HashScheme((0, 1), 16)
    assert h1 != HashScheme((0, 1), 32)
    assert h1 != HashScheme((2,), 16)


def test_order_scheme() -> None:
    """Test OrderScheme construction, properties, equality, and repr."""
    o1 = OrderScheme(
        column_indices=(0, 1),
        orders=("ascending", "descending"),
        null_orders=("first", "last"),
    )
    assert o1.column_indices == (0, 1)
    assert o1.orders == ("ascending", "descending")
    assert o1.null_orders == ("first", "last")
    assert not o1.has_boundaries
    assert o1.get_boundaries_table() is None
    assert "OrderScheme" in repr(o1)
    assert "ascending" in repr(o1)

    # Equality (without boundaries)
    o2 = OrderScheme(
        column_indices=(0, 1),
        orders=("ascending", "descending"),
        null_orders=("first", "last"),
    )
    assert o1 == o2

    # Different orders
    o3 = OrderScheme(
        column_indices=(0, 1),
        orders=("descending", "descending"),
        null_orders=("first", "last"),
    )
    assert o1 != o3

    # Different column indices
    o4 = OrderScheme(
        column_indices=(2,),
        orders=("ascending",),
        null_orders=("first",),
    )
    assert o1 != o4


def test_order_scheme_with_boundaries() -> None:
    """Test OrderScheme with boundaries TableChunk (multi-column)."""
    # Create boundaries table with 2 columns (for composite sort key)
    # 2 boundary rows for 3 partitions
    df = cudf.DataFrame(
        {
            "key1": [100, 200],
            "key2": ["abc", "xyz"],
        }
    )
    stream = rmm.pylibrmm.stream.DEFAULT_STREAM
    boundaries = TableChunk.from_pylibcudf_table(
        cudf_to_pylibcudf_table(df), stream, exclusive_view=False
    )

    o1 = OrderScheme(
        column_indices=(0, 1),
        orders=("ascending", "descending"),
        null_orders=("first", "last"),
        boundaries=boundaries,
    )
    assert o1.column_indices == (0, 1)
    assert o1.orders == ("ascending", "descending")
    assert o1.null_orders == ("first", "last")
    assert o1.has_boundaries
    # Get the boundaries as a pylibcudf.Table
    tbl = o1.get_boundaries_table()
    assert tbl is not None
    assert tbl.num_columns() == 2
    assert tbl.num_rows() == 2


def test_order_scheme_validation() -> None:
    """Test OrderScheme input validation."""
    # Mismatched lengths
    with pytest.raises(ValueError, match="same length"):
        OrderScheme(
            column_indices=(0, 1),
            orders=("ascending",),  # Wrong length
            null_orders=("first", "last"),
        )

    with pytest.raises(ValueError, match="same length"):
        OrderScheme(
            column_indices=(0, 1),
            orders=("ascending", "descending"),
            null_orders=("first",),  # Wrong length
        )

    # Invalid order string
    with pytest.raises(ValueError, match="Invalid order"):
        OrderScheme(
            column_indices=(0,),
            orders=("invalid",),  # type: ignore[arg-type]
            null_orders=("first",),
        )

    # Invalid null_order string
    with pytest.raises(ValueError, match="Invalid null_order"):
        OrderScheme(
            column_indices=(0,),
            orders=("ascending",),
            null_orders=("invalid",),  # type: ignore[arg-type]
        )


def test_partitioning_scenarios() -> None:
    """Test various partitioning configurations."""
    # Default / None
    p_default = Partitioning()
    assert p_default.inter_rank is None
    assert p_default.local is None
    assert Partitioning(None, None) == p_default

    # Direct global shuffle: inter_rank=Hash, local=Aligned
    p_global = Partitioning(HashScheme((0,), 16), "inherit")
    assert p_global.inter_rank == HashScheme((0,), 16)
    assert p_global.local == "inherit"

    # Two-stage shuffle: inter_rank=Hash(nranks), local=Hash(N_l)
    p_twostage = Partitioning(HashScheme((0,), 4), HashScheme((0,), 8))
    assert p_twostage.inter_rank == HashScheme((0,), 4)
    assert p_twostage.local == HashScheme((0,), 8)

    # After local repartition: inter_rank=Hash, local=None
    p_local_none = Partitioning(HashScheme((0,), 16), None)
    assert p_local_none.inter_rank == HashScheme((0,), 16)
    assert p_local_none.local is None

    # Order-based partitioning (range partitioned / sorted)
    order_scheme = OrderScheme(
        column_indices=(0,),
        orders=("ascending",),
        null_orders=("last",),
    )
    p_ordered = Partitioning(order_scheme, "inherit")
    assert p_ordered.inter_rank == order_scheme
    assert p_ordered.local == "inherit"

    # Mixed: inter_rank=Order, local=Hash
    p_mixed = Partitioning(
        OrderScheme((0,), ("ascending",), ("first",)),
        HashScheme((1,), 8),
    )
    assert isinstance(p_mixed.inter_rank, OrderScheme)
    assert isinstance(p_mixed.local, HashScheme)

    # Equality and repr
    assert p_global == Partitioning(HashScheme((0,), 16), "inherit")
    assert p_global != p_twostage
    assert "Partitioning" in repr(p_global)
    assert "inter_rank" in repr(p_global)

    # Invalid type
    with pytest.raises(TypeError):
        Partitioning("invalid", None)  # type: ignore[arg-type]


def test_channel_metadata() -> None:
    """Test ChannelMetadata construction and properties."""
    # Basic construction
    m = ChannelMetadata(local_count=4)
    assert m.local_count == 4
    assert m.duplicated is False

    # With partitioning and duplicated
    p = Partitioning(HashScheme((0,), 16), "inherit")
    m_full = ChannelMetadata(local_count=4, partitioning=p, duplicated=True)
    assert m_full.partitioning == p
    assert m_full.duplicated is True

    # Equality and repr
    m2 = ChannelMetadata(local_count=4)
    assert m == m2
    assert m != ChannelMetadata(local_count=8)
    assert "local_count=4" in repr(m)

    # Validation
    with pytest.raises(ValueError, match="local_count must be non-negative"):
        ChannelMetadata(local_count=-1)


def test_message_roundtrip() -> None:
    """Test ChannelMetadata can round-trip through Message."""
    m = ChannelMetadata(
        local_count=4,
        partitioning=Partitioning(HashScheme((0,), 16), "inherit"),
        duplicated=True,
    )
    msg_m = Message(99, m)
    assert msg_m.sequence_number == 99
    got_m = ChannelMetadata.from_message(msg_m)
    assert got_m.local_count == 4
    assert got_m.duplicated is True
    assert got_m.partitioning.inter_rank == HashScheme((0,), 16)
    assert msg_m.empty()


def test_message_roundtrip_with_order_scheme() -> None:
    """Test ChannelMetadata with OrderScheme can round-trip through Message."""
    # Create boundaries TableChunk
    df = cudf.DataFrame({"key1": [100, 200], "key2": ["abc", "xyz"]})
    stream = rmm.pylibrmm.stream.DEFAULT_STREAM
    boundaries = TableChunk.from_pylibcudf_table(
        cudf_to_pylibcudf_table(df), stream, exclusive_view=False
    )

    order_scheme = OrderScheme(
        column_indices=(0, 1),
        orders=("ascending", "descending"),
        null_orders=("first", "last"),
        boundaries=boundaries,
    )
    assert order_scheme.has_boundaries

    m = ChannelMetadata(
        local_count=8,
        partitioning=Partitioning(order_scheme, "inherit"),
        duplicated=True,
    )
    msg_m = Message(42, m)
    assert msg_m.sequence_number == 42
    got_m = ChannelMetadata.from_message(msg_m)
    assert got_m.local_count == 8
    assert got_m.duplicated is True
    assert isinstance(got_m.partitioning.inter_rank, OrderScheme)
    assert got_m.partitioning.inter_rank.column_indices == (0, 1)
    assert got_m.partitioning.inter_rank.orders == ("ascending", "descending")
    assert got_m.partitioning.inter_rank.null_orders == ("first", "last")
    assert got_m.partitioning.local == "inherit"
    # Boundaries should round-trip through the message
    assert got_m.partitioning.inter_rank.has_boundaries
    tbl = got_m.partitioning.inter_rank.get_boundaries_table()
    assert tbl is not None
    assert tbl.num_columns() == 2
    assert tbl.num_rows() == 2
    assert msg_m.empty()


def test_access_after_move_raises() -> None:
    """Test that accessing a released ChannelMetadata raises ValueError."""
    m = ChannelMetadata(
        local_count=4,
        partitioning=Partitioning(HashScheme((0,), 16), "inherit"),
    )
    # Move into a message (releases the handle)
    _ = Message(0, m)

    # Accessing any property should raise ValueError
    with pytest.raises(ValueError, match="uninitialized"):
        _ = m.local_count

    with pytest.raises(ValueError, match="uninitialized"):
        _ = m.partitioning

    with pytest.raises(ValueError, match="uninitialized"):
        _ = m.duplicated

    with pytest.raises(ValueError, match="uninitialized"):
        repr(m)
