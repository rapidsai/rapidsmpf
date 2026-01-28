# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Type stubs for channel_metadata module."""

from __future__ import annotations

from typing import Literal

from rapidsmpf.streaming.core.message import Message

class HashScheme:
    """
    Hash partitioning scheme.

    Rows are distributed by ``hash(table[column_indices]) % modulus``.
    """

    def __init__(self, column_indices: tuple[int, ...], modulus: int) -> None: ...
    @property
    def column_indices(self) -> tuple[int, ...]: ...
    @property
    def modulus(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...

PartitioningSpecValue = HashScheme | None | Literal["aligned"]

class Partitioning:
    """
    Hierarchical partitioning metadata for a data stream.

    Describes how data flowing through a channel is partitioned at multiple
    levels of the system hierarchy.
    """

    def __init__(
        self,
        inter_rank: PartitioningSpecValue = None,
        local: PartitioningSpecValue = None,
    ) -> None: ...
    @property
    def inter_rank(self) -> PartitioningSpecValue: ...
    @property
    def local(self) -> PartitioningSpecValue: ...
    def __eq__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...

class ChannelMetadata:
    """
    Channel-level metadata describing the data stream.

    Contains information about chunk counts, partitioning, and duplication
    status for the data flowing through a channel.
    """

    def __init__(
        self,
        local_count: int,
        *,
        partitioning: Partitioning | None = None,
        duplicated: bool = False,
    ) -> None: ...
    @staticmethod
    def from_message(message: Message) -> ChannelMetadata: ...
    def into_message(self, sequence_number: int, message: Message) -> None: ...
    @property
    def local_count(self) -> int: ...
    @property
    def partitioning(self) -> Partitioning: ...
    @property
    def duplicated(self) -> bool: ...
    def __eq__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
