# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Tests for streaming metadata types (Partitioning and ChannelMetadata)."""

from __future__ import annotations

import pytest

from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.metadata import ChannelMetadata, HashScheme, Partitioning

# ============================================================================
# HashScheme Tests
# ============================================================================


def test_hash_scheme_construction() -> None:
    h = HashScheme(("col_a", "col_b"), 16)
    assert h.columns == ("col_a", "col_b")
    assert h.modulus == 16


def test_hash_scheme_single_column() -> None:
    h = HashScheme(("key",), 32)
    assert h.columns == ("key",)
    assert h.modulus == 32


def test_hash_scheme_equality() -> None:
    h1 = HashScheme(("key",), 16)
    h2 = HashScheme(("key",), 16)
    h3 = HashScheme(("key",), 32)
    h4 = HashScheme(("other",), 16)

    assert h1 == h2
    assert h1 != h3
    assert h1 != h4


def test_hash_scheme_repr() -> None:
    h = HashScheme(("key",), 16)
    assert repr(h) == "HashScheme(('key',), 16)"


# ============================================================================
# Partitioning Tests
# ============================================================================


def test_partitioning_default() -> None:
    p = Partitioning()
    assert p.inter_rank is None
    assert p.local is None


def test_partitioning_none_values() -> None:
    p = Partitioning(None, None)
    assert p.inter_rank is None
    assert p.local is None


def test_partitioning_direct_global_shuffle() -> None:
    """Direct global shuffle to N_g partitions: inter_rank=Hash, local=Aligned."""
    p = Partitioning(HashScheme(("key",), 16), "aligned")
    assert isinstance(p.inter_rank, HashScheme)
    assert p.inter_rank.columns == ("key",)
    assert p.inter_rank.modulus == 16
    assert p.local == "aligned"


def test_partitioning_two_stage_shuffle() -> None:
    """Two-stage shuffle: inter_rank=Hash(nranks), local=Hash(N_l)."""
    p = Partitioning(HashScheme(("key",), 4), HashScheme(("key",), 8))
    assert isinstance(p.inter_rank, HashScheme)
    assert isinstance(p.local, HashScheme)
    assert p.inter_rank.modulus == 4
    assert p.local.modulus == 8


def test_partitioning_after_local_repartition() -> None:
    """After local repartition: inter_rank=Hash, local=None."""
    p = Partitioning(HashScheme(("key",), 16), None)
    assert isinstance(p.inter_rank, HashScheme)
    assert p.local is None


def test_partitioning_equality() -> None:
    p1 = Partitioning(HashScheme(("key",), 16), "aligned")
    p2 = Partitioning(HashScheme(("key",), 16), "aligned")
    p3 = Partitioning(HashScheme(("key",), 32), "aligned")

    assert p1 == p2
    assert p1 != p3


def test_partitioning_repr() -> None:
    p = Partitioning(HashScheme(("key",), 16), "aligned")
    assert "Partitioning" in repr(p)
    assert "inter_rank" in repr(p)
    assert "local" in repr(p)


def test_partitioning_invalid_type() -> None:
    with pytest.raises(TypeError):
        Partitioning("invalid", None)  # type: ignore[arg-type]


# ============================================================================
# ChannelMetadata Tests
# ============================================================================


def test_channel_metadata_construction() -> None:
    m = ChannelMetadata(local_count=4, global_count=16)
    assert m.local_count == 4
    assert m.global_count == 16
    assert m.duplicated is False


def test_channel_metadata_no_global_count() -> None:
    m = ChannelMetadata(local_count=4)
    assert m.local_count == 4
    assert m.global_count is None


def test_channel_metadata_with_partitioning() -> None:
    p = Partitioning(HashScheme(("key",), 16), "aligned")
    m = ChannelMetadata(local_count=4, global_count=16, partitioning=p)
    assert m.partitioning == p


def test_channel_metadata_duplicated() -> None:
    m = ChannelMetadata(local_count=1, duplicated=True)
    assert m.duplicated is True


def test_channel_metadata_equality() -> None:
    p = Partitioning(HashScheme(("key",), 16), "aligned")
    m1 = ChannelMetadata(local_count=4, global_count=16, partitioning=p)
    m2 = ChannelMetadata(local_count=4, global_count=16, partitioning=p)
    m3 = ChannelMetadata(local_count=8, global_count=16, partitioning=p)

    assert m1 == m2
    assert m1 != m3


def test_channel_metadata_repr() -> None:
    m = ChannelMetadata(local_count=4, global_count=16)
    r = repr(m)
    assert "ChannelMetadata" in r
    assert "local_count=4" in r
    assert "global_count=16" in r


def test_channel_metadata_negative_local_count() -> None:
    with pytest.raises(ValueError, match="local_count must be non-negative"):
        ChannelMetadata(local_count=-1)


def test_channel_metadata_negative_global_count() -> None:
    with pytest.raises(ValueError, match="global_count must be non-negative"):
        ChannelMetadata(local_count=4, global_count=-1)


# ============================================================================
# Message Integration Tests
# ============================================================================


def test_partitioning_roundtrip_message() -> None:
    """Test that Partitioning can be wrapped in a Message and extracted."""
    p = Partitioning(HashScheme(("key",), 16), "aligned")
    msg = Message(42, p)
    assert msg.sequence_number == 42

    got = Partitioning.from_message(msg)
    assert got == p
    assert msg.empty()


def test_channel_metadata_roundtrip_message() -> None:
    """Test that ChannelMetadata can be wrapped in a Message and extracted."""
    p = Partitioning(HashScheme(("key",), 16), "aligned")
    m = ChannelMetadata(
        local_count=4, global_count=16, partitioning=p, duplicated=False
    )
    msg = Message(42, m)
    assert msg.sequence_number == 42

    got = ChannelMetadata.from_message(msg)
    assert got.local_count == 4
    assert got.global_count == 16
    assert got.partitioning == p
    assert got.duplicated is False
    assert msg.empty()
