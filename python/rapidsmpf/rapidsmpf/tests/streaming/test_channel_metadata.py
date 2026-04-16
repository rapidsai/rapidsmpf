# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Tests for streaming metadata types (Partitioning and ChannelMetadata)."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

import cudf
import pylibcudf as plc

from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.cudf import (
    ChannelMetadata,
    HashScheme,
    OrderKey,
    OrderScheme,
    Partitioning,
    TableChunk,
)
from rapidsmpf.utils.cudf import cudf_to_pylibcudf_table

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rapidsmpf.streaming.core.context import Context


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


def _two_key_order_scheme(*, strict_boundary: bool = False) -> OrderScheme:
    return OrderScheme(
        [
            OrderKey(0, plc.types.Order.ASCENDING, plc.types.NullOrder.BEFORE),
            OrderKey(1, plc.types.Order.DESCENDING, plc.types.NullOrder.AFTER),
        ],
        strict_boundary=strict_boundary,
    )


def test_order_key() -> None:
    k = OrderKey(0, plc.types.Order.ASCENDING, plc.types.NullOrder.BEFORE)
    assert k.column_index == 0
    assert k.order == plc.types.Order.ASCENDING
    assert k.null_order == plc.types.NullOrder.BEFORE
    assert k == OrderKey(0, plc.types.Order.ASCENDING, plc.types.NullOrder.BEFORE)
    assert k != OrderKey(1, plc.types.Order.ASCENDING, plc.types.NullOrder.BEFORE)
    assert "OrderKey" in repr(k)

    with pytest.raises(ValueError, match="Invalid order"):
        OrderKey(0, 99, plc.types.NullOrder.BEFORE)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Invalid null order"):
        OrderKey(0, plc.types.Order.ASCENDING, 99)  # type: ignore[arg-type]


def test_order_scheme() -> None:
    """Test OrderScheme construction, properties, equality, and repr."""
    o1 = _two_key_order_scheme()
    assert o1.keys == (
        OrderKey(0, plc.types.Order.ASCENDING, plc.types.NullOrder.BEFORE),
        OrderKey(1, plc.types.Order.DESCENDING, plc.types.NullOrder.AFTER),
    )
    assert o1.strict_boundary is False
    assert o1.get_boundaries_table() is None
    assert "OrderScheme" in repr(o1)

    # Equality
    assert o1 == _two_key_order_scheme()
    assert o1 != OrderScheme(
        [OrderKey(0, plc.types.Order.DESCENDING, plc.types.NullOrder.BEFORE)]
    )
    assert o1 != OrderScheme(
        [OrderKey(2, plc.types.Order.ASCENDING, plc.types.NullOrder.BEFORE)]
    )

    o_strict = _two_key_order_scheme(strict_boundary=True)
    assert o_strict.strict_boundary is True
    assert o1 != o_strict
    assert o_strict == _two_key_order_scheme(strict_boundary=True)

    with pytest.raises(TypeError, match="OrderKey"):
        OrderScheme(
            cast(
                "Sequence[OrderKey]",
                [(0, plc.types.Order.ASCENDING, plc.types.NullOrder.BEFORE)],
            )
        )

    with pytest.raises(ValueError, match="empty"):
        OrderScheme([])


def test_order_scheme_with_boundaries(context: Context) -> None:
    """Test OrderScheme with boundaries TableChunk (multi-column)."""
    df = cudf.DataFrame({"key1": [100, 200], "key2": ["abc", "xyz"]})
    stream = context.get_stream_from_pool()
    boundaries = TableChunk.from_pylibcudf_table(
        cudf_to_pylibcudf_table(df),
        stream,
        exclusive_view=False,
        br=context.br(),
    )
    o1 = OrderScheme(
        [
            OrderKey(0, plc.types.Order.ASCENDING, plc.types.NullOrder.BEFORE),
            OrderKey(1, plc.types.Order.DESCENDING, plc.types.NullOrder.AFTER),
        ],
        boundaries=boundaries,
    )
    assert o1.num_boundaries == 2
    tbl = o1.get_boundaries_table()
    assert tbl is not None
    assert tbl.num_columns() == 2
    assert tbl.num_rows() == 2


def test_num_boundaries() -> None:
    """num_boundaries is None without boundaries, returns row count otherwise."""
    o_no_b = _two_key_order_scheme()
    assert o_no_b.num_boundaries is None


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
        [OrderKey(0, plc.types.Order.ASCENDING, plc.types.NullOrder.AFTER)]
    )
    p_ordered = Partitioning(order_scheme, "inherit")
    assert p_ordered.inter_rank == order_scheme
    assert p_ordered.local == "inherit"

    # Mixed: inter_rank=Order, local=Hash
    p_mixed = Partitioning(
        OrderScheme(
            [OrderKey(0, plc.types.Order.ASCENDING, plc.types.NullOrder.BEFORE)]
        ),
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


def test_message_roundtrip_with_order_scheme(context: Context) -> None:
    """Test ChannelMetadata with OrderScheme can round-trip through Message."""
    df = cudf.DataFrame({"key1": [100, 200], "key2": ["abc", "xyz"]})
    stream = context.get_stream_from_pool()
    boundaries = TableChunk.from_pylibcudf_table(
        cudf_to_pylibcudf_table(df),
        stream,
        exclusive_view=False,
        br=context.br(),
    )
    order_scheme = OrderScheme(
        [
            OrderKey(0, plc.types.Order.ASCENDING, plc.types.NullOrder.BEFORE),
            OrderKey(1, plc.types.Order.DESCENDING, plc.types.NullOrder.AFTER),
        ],
        boundaries=boundaries,
        strict_boundary=True,
    )
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
    assert got_m.partitioning.inter_rank.keys == (
        OrderKey(0, plc.types.Order.ASCENDING, plc.types.NullOrder.BEFORE),
        OrderKey(1, plc.types.Order.DESCENDING, plc.types.NullOrder.AFTER),
    )
    assert got_m.partitioning.local == "inherit"
    assert got_m.partitioning.inter_rank.strict_boundary is True
    assert got_m.partitioning.inter_rank.num_boundaries == 2
    tbl = got_m.partitioning.inter_rank.get_boundaries_table()
    assert tbl is not None
    assert tbl.num_columns() == 2
    assert tbl.num_rows() == 2
    assert msg_m.empty()


def test_order_scheme_view_roundtrip() -> None:
    """Re-feeding a view OrderScheme (from metadata.partitioning.inter_rank) into
    Partitioning must use the live cpp_OrderScheme, not the wrapper's empty slot."""
    src = ChannelMetadata(
        local_count=1,
        partitioning=Partitioning(_two_key_order_scheme(), "inherit"),
    )
    # inter_rank is a non-owning view into src's storage
    view_scheme = src.partitioning.inter_rank
    assert isinstance(view_scheme, OrderScheme)

    # Round-trip the view through a new Partitioning
    p2 = Partitioning(view_scheme, None)
    assert p2.inter_rank == _two_key_order_scheme()


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
