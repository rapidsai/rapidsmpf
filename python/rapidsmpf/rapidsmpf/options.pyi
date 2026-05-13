# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")

@dataclass(frozen=True)
class OptionDescriptor(Generic[T]):
    key: str
    default_value: T

class statistics:
    EnabledOption: OptionDescriptor[str]

class pinned_memory:
    EnabledOption: OptionDescriptor[bool]
    InitialPoolSizeFactorOption: OptionDescriptor[str]
    MaxPoolSizeFactorOption: OptionDescriptor[str]

class buffer_resource:
    SpillDeviceLimitOption: OptionDescriptor[str]
    PeriodicSpillCheckOption: OptionDescriptor[str]
    NumStreamsOption: OptionDescriptor[int]

class streaming:
    NumStreamingThreadsOption: OptionDescriptor[int]
    MemoryReserveTimeoutOption: OptionDescriptor[str]

class communicator:
    LogOption: OptionDescriptor[str]

class ucxx:
    ProgressModeOption: OptionDescriptor[str]
