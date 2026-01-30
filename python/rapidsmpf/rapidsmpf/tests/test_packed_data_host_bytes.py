# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Tests for PackedData host bytes functionality."""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import rmm

from rapidsmpf.memory.buffer_resource import BufferResource
from rapidsmpf.memory.packed_data import PackedData


def test_packed_data_host_bytes_roundtrip(
    device_mr: rmm.mr.CudaMemoryResource,
) -> None:
    """Test creating PackedData from host bytes and extracting them back."""
    br = BufferResource(device_mr)

    # Test with raw bytes
    original = b"hello world"
    packed = PackedData.from_host_bytes(original, br)
    result = packed.to_host_bytes()
    assert result == original

    # Test with struct-packed int64
    value = 12345678901234
    original = struct.pack("q", value)
    packed = PackedData.from_host_bytes(original, br)
    result = packed.to_host_bytes()
    assert struct.unpack("q", result)[0] == value


def test_packed_data_empty_bytes(
    device_mr: rmm.mr.CudaMemoryResource,
) -> None:
    """Test creating PackedData from empty bytes."""
    br = BufferResource(device_mr)
    original = b""
    packed = PackedData.from_host_bytes(original, br)
    result = packed.to_host_bytes()
    assert result == original
