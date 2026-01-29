# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Tests for streaming metadata types (Partitioning and ChannelMetadata)."""

from __future__ import annotations

import pytest

from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.cudf import ChannelMetadata, HashScheme, Partitioning


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


def test_partitioning_scenarios() -> None:
    """Test various partitioning configurations."""
    # Default / None
    p_default = Partitioning()
    assert p_default.inter_rank is None
    assert p_default.local is None
    assert Partitioning(None, None) == p_default

    # Direct global shuffle: inter_rank=Hash, local=Aligned
    p_global = Partitioning(HashScheme((0,), 16), "passthrough")
    assert p_global.inter_rank == HashScheme((0,), 16)
    assert p_global.local == "passthrough"

    # Two-stage shuffle: inter_rank=Hash(nranks), local=Hash(N_l)
    p_twostage = Partitioning(HashScheme((0,), 4), HashScheme((0,), 8))
    assert p_twostage.inter_rank == HashScheme((0,), 4)
    assert p_twostage.local == HashScheme((0,), 8)

    # After local repartition: inter_rank=Hash, local=None
    p_local_none = Partitioning(HashScheme((0,), 16), None)
    assert p_local_none.inter_rank == HashScheme((0,), 16)
    assert p_local_none.local is None

    # Equality and repr
    assert p_global == Partitioning(HashScheme((0,), 16), "passthrough")
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
    p = Partitioning(HashScheme((0,), 16), "passthrough")
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
        partitioning=Partitioning(HashScheme((0,), 16), "passthrough"),
        duplicated=True,
    )
    msg_m = Message(99, m)
    assert msg_m.sequence_number == 99
    got_m = ChannelMetadata.from_message(msg_m)
    assert got_m.local_count == 4
    assert got_m.duplicated is True
    assert got_m.partitioning.inter_rank == HashScheme((0,), 16)
    assert msg_m.empty()


def test_access_after_move_raises() -> None:
    """Test that accessing a released ChannelMetadata raises ValueError."""
    m = ChannelMetadata(
        local_count=4,
        partitioning=Partitioning(HashScheme((0,), 16), "passthrough"),
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
