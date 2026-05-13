# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from typing import Generic, NamedTuple, TypeVar

T = TypeVar("T")

class OptionDescriptor(NamedTuple, Generic[T]):
    key: str
    default_val: T

StatisticsEnabledOption: OptionDescriptor[str]

PinnedMemoryEnabledOption: OptionDescriptor[bool]
PinnedMemoryInitialPoolSizeFactorOption: OptionDescriptor[str]
PinnedMemoryMaxPoolSizeFactorOption: OptionDescriptor[str]

BufferResourceSpillDeviceLimitOption: OptionDescriptor[str]
BufferResourcePeriodicSpillCheckOption: OptionDescriptor[str]
BufferResourceNumStreamsOption: OptionDescriptor[int]

StreamingNumStreamingThreadsOption: OptionDescriptor[int]
StreamingMemoryReserveTimeoutOption: OptionDescriptor[str]

CommunicatorLogOption: OptionDescriptor[str]

UcxxProgressModeOption: OptionDescriptor[str]

__all__: list[str]
