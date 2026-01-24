# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Tests for streaming metadata types (Partitioning and ChannelMetadata)."""

from __future__ import annotations

import pytest

from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.cudf import ChannelMetadata, HashScheme, Partitioning


def test_hash_scheme() -> None:
    """Test HashScheme construction, properties, equality, and repr."""
    h1 = HashScheme(("col_a", "col_b"), 16)
    assert h1.columns == ("col_a", "col_b")
    assert h1.modulus == 16
    assert repr(h1) == "HashScheme(('col_a', 'col_b'), 16)"

    # Equality
    assert h1 == HashScheme(("col_a", "col_b"), 16)
    assert h1 != HashScheme(("col_a", "col_b"), 32)
    assert h1 != HashScheme(("other",), 16)


def test_partitioning_scenarios() -> None:
    """Test various partitioning configurations."""
    # Default / None
    p_default = Partitioning()
    assert p_default.inter_rank is None
    assert p_default.local is None
    assert Partitioning(None, None) == p_default

    # Direct global shuffle: inter_rank=Hash, local=Aligned
    p_global = Partitioning(HashScheme(("key",), 16), "aligned")
    assert p_global.inter_rank == HashScheme(("key",), 16)
    assert p_global.local == "aligned"

    # Two-stage shuffle: inter_rank=Hash(nranks), local=Hash(N_l)
    p_twostage = Partitioning(HashScheme(("key",), 4), HashScheme(("key",), 8))
    assert p_twostage.inter_rank == HashScheme(("key",), 4)
    assert p_twostage.local == HashScheme(("key",), 8)

    # After local repartition: inter_rank=Hash, local=None
    p_local_none = Partitioning(HashScheme(("key",), 16), None)
    assert p_local_none.inter_rank == HashScheme(("key",), 16)
    assert p_local_none.local is None

    # Equality and repr
    assert p_global == Partitioning(HashScheme(("key",), 16), "aligned")
    assert p_global != p_twostage
    assert "Partitioning" in repr(p_global)
    assert "inter_rank" in repr(p_global)

    # Invalid type
    with pytest.raises(TypeError):
        Partitioning("invalid", None)  # type: ignore[arg-type]


def test_channel_metadata() -> None:
    """Test ChannelMetadata construction and properties."""
    # Basic construction
    m = ChannelMetadata(local_count=4, global_count=16)
    assert m.local_count == 4
    assert m.global_count == 16
    assert m.duplicated is False

    # Optional global_count
    assert ChannelMetadata(local_count=4).global_count is None

    # With partitioning and duplicated
    p = Partitioning(HashScheme(("key",), 16), "aligned")
    m_full = ChannelMetadata(
        local_count=4, global_count=16, partitioning=p, duplicated=True
    )
    assert m_full.partitioning == p
    assert m_full.duplicated is True

    # Equality and repr
    m2 = ChannelMetadata(local_count=4, global_count=16)
    assert m == m2
    assert m != ChannelMetadata(local_count=8, global_count=16)
    assert "local_count=4" in repr(m)

    # Validation
    with pytest.raises(ValueError, match="local_count must be non-negative"):
        ChannelMetadata(local_count=-1)
    with pytest.raises(ValueError, match="global_count must be non-negative"):
        ChannelMetadata(local_count=4, global_count=-1)


def test_message_roundtrip() -> None:
    """Test Partitioning and ChannelMetadata can round-trip through Message."""
    # Partitioning round-trip
    p = Partitioning(HashScheme(("key",), 16), "aligned")
    msg_p = Message(42, p)
    assert msg_p.sequence_number == 42
    got_p = Partitioning.from_message(msg_p)
    assert got_p.inter_rank == HashScheme(("key",), 16)
    assert got_p.local == "aligned"
    assert msg_p.empty()

    # ChannelMetadata round-trip
    m = ChannelMetadata(
        local_count=4,
        global_count=16,
        partitioning=Partitioning(HashScheme(("key",), 16), "aligned"),
        duplicated=True,
    )
    msg_m = Message(99, m)
    got_m = ChannelMetadata.from_message(msg_m)
    assert got_m.local_count == 4
    assert got_m.global_count == 16
    assert got_m.duplicated is True
    assert got_m.partitioning.inter_rank == HashScheme(("key",), 16)
    assert msg_m.empty()
