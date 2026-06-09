# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Submodule for testing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pylibcudf

import rmm
from rmm.pylibrmm.stream import DEFAULT_STREAM

from rapidsmpf.memory.packed_data import PackedData

if TYPE_CHECKING:
    from rmm.pylibrmm.stream import Stream

    from rapidsmpf.memory.buffer_resource import BufferResource

# Element type stored in the synthetic packed payloads. Values are wide enough
# that any per-shuffle ``base`` offsets used by callers never collide.
_DTYPE = np.int64


def assert_eq(
    left: pylibcudf.Table,
    right: pylibcudf.Table,
    *,
    sort_rows: int | None = None,
    stream: Stream | None = None,
) -> None:
    """
    Assert that two tables are equivalent using pylibcudf.

    Parameters
    ----------
    left
        plc.Table to compare.
    right
        plc.Table to compare.
    sort_rows
        If not None, sort both tables by this column before comparing.
        An ``int`` is treated as a column index.
    stream
        CUDA stream to use for the comparison.

    Raises
    ------
    AssertionError
        If the two tables do not compare equal.
    """
    if stream is None:
        stream = DEFAULT_STREAM

    if sort_rows is not None:
        column_order = [pylibcudf.types.Order.ASCENDING]
        null_precedence = [pylibcudf.types.NullOrder.BEFORE]
        left = pylibcudf.sorting.stable_sort_by_key(
            left,
            pylibcudf.Table([left.columns()[sort_rows]]),
            column_order,
            null_precedence,
            stream=stream,
        )
        right = pylibcudf.sorting.stable_sort_by_key(
            right,
            pylibcudf.Table([right.columns()[sort_rows]]),
            column_order,
            null_precedence,
            stream=stream,
        )
    if not pylibcudf.table_equality.tables_equal(left, right, stream=stream):
        raise AssertionError(f"Table are not equal with {sort_rows=}")


def chunk_indices(count: int, num_chunks: int) -> list[tuple[int, int]]:
    """
    Split ``[0, count)`` into ``num_chunks`` contiguous, front-loaded pieces.

    Mirrors ``rapidsmpf::chunk_indices``: when ``count < num_chunks`` the trailing
    pieces are empty (``start == end``). The returned pieces exactly tile
    ``[0, count)``.

    Parameters
    ----------
    count
        Size of the range to split.
    num_chunks
        Number of pieces to split the range into.

    Returns
    -------
    A list of ``(start, end)`` pairs, one per chunk.
    """
    chunk_size = -(-count // num_chunks)  # ceil division
    return [
        (min(k * chunk_size, count), min((k + 1) * chunk_size, count))
        for k in range(num_chunks)
    ]


def generate_packed_data(
    n_elements: int, offset: int, stream: Stream, br: BufferResource
) -> PackedData:
    """
    Build a ``PackedData`` holding the int sequence ``[offset, offset + n_elements)``.

    The sequence is stored as a device buffer payload (and mirrored in the metadata)
    so it survives a shuffle round-trip and can be validated by
    :func:`validate_packed_data`.

    Parameters
    ----------
    n_elements
        Number of elements in the sequence.
    offset
        Starting value of the sequence.
    stream
        CUDA stream used for the device allocation.
    br
        Buffer resource used for memory management.

    Returns
    -------
    A ``PackedData`` containing the integer sequence.
    """
    data = np.arange(offset, offset + n_elements, dtype=_DTYPE).tobytes()
    gpu_data = rmm.DeviceBuffer.to_device(data, stream=stream)
    return PackedData.from_device_buffer(gpu_data, data, stream, br)


def validate_packed_data(packed_data: PackedData, n_elements: int, offset: int) -> None:
    """
    Check that ``packed_data`` holds the sequence ``[offset, offset + n_elements)``.

    Parameters
    ----------
    packed_data
        Packed data to validate.
    n_elements
        Expected number of elements.
    offset
        Expected starting value of the sequence.
    """
    values = np.frombuffer(packed_data.to_host_bytes(), dtype=_DTYPE)
    np.testing.assert_array_equal(
        values, np.arange(offset, offset + n_elements, dtype=_DTYPE)
    )


def make_partition_data(
    total_num_partitions: int,
    total_num_rows: int,
    local_pid: int,
    stream: Stream,
    br: BufferResource,
    base: int = 0,
) -> dict[int, PackedData]:
    """
    Produce the non-empty sub-regions of one owned input region ``local_pid``.

    The index range ``[0, total_num_rows)`` is split into ``P * P`` contiguous
    sub-regions (``P == total_num_partitions``). Sub-region ``(local_pid,
    split_idx)`` is piece ``local_pid * P + split_idx`` and is routed to
    destination partition ``split_idx``. Since ``local_partitions()`` across ranks
    partition ``[0, P)``, every input region is produced exactly once, so rows are
    not replicated and the total shuffled data equals ``total_num_rows``.

    Parameters
    ----------
    total_num_partitions
        Total number of partitions in the shuffle.
    total_num_rows
        Total number of rows tiled across all input regions.
    local_pid
        Index of the owned input region to produce.
    stream
        CUDA stream used for device allocations.
    br
        Buffer resource used for memory management.
    base
        Offset added to every value so distinct shuffles carry distinct data.

    Returns
    -------
    A map of destination partition ID to its packed sub-region.
    """
    P = total_num_partitions
    pieces = chunk_indices(total_num_rows, P * P)

    chunks: dict[int, PackedData] = {}
    for split_idx in range(P):
        start, end = pieces[local_pid * P + split_idx]
        if end > start:
            chunks[split_idx] = generate_packed_data(
                end - start, base + start, stream, br
            )
    return chunks


def validate_partition_data(
    received: list[PackedData],
    total_num_partitions: int,
    total_num_rows: int,
    local_pid: int,
    base: int = 0,
) -> None:
    """
    Verify that the ``received`` chunks for partition ``local_pid`` are as expected.

    Checks the chunks against the non-empty sub-regions expected for partition
    ``local_pid`` under the conserved, front-loaded data model of
    :func:`make_partition_data`.

    Parameters
    ----------
    received
        The packed data chunks extracted for partition ``local_pid``.
    total_num_partitions
        Total number of partitions in the shuffle.
    total_num_rows
        Total number of rows tiled across all input regions.
    local_pid
        Partition ID being validated.
    base
        Offset that was added to every value when the data was generated.
    """
    P = total_num_partitions
    pieces = chunk_indices(total_num_rows, P * P)

    # Recompute the non-empty (offset, count) sub-regions expected for partition
    # local_pid, in increasing input-region-index (== increasing offset) order.
    expected: list[tuple[int, int]] = []
    for i in range(P):
        start, end = pieces[i * P + local_pid]
        if end > start:
            expected.append((base + start, end - start))

    assert len(received) == len(expected)

    # Decode each received chunk and sort by its first element (== offset) so they
    # align 1:1 with the expected list, which is already in offset order.
    decoded = sorted(
        (np.frombuffer(pd.to_host_bytes(), dtype=_DTYPE) for pd in received),
        key=lambda arr: int(arr[0]),
    )
    for arr, (off, cnt) in zip(decoded, expected, strict=True):
        np.testing.assert_array_equal(arr, np.arange(off, off + cnt, dtype=_DTYPE))
