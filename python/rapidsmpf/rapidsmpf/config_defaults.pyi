# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Mapping
from typing import Final

STATISTICS_ENABLED: Final[str]

PINNED_MEMORY_ENABLED: Final[str]
PINNED_MEMORY_INITIAL_POOL_SIZE: Final[str]
PINNED_MEMORY_MAX_POOL_SIZE: Final[str]

BUFFER_RESOURCE_SPILL_DEVICE_LIMIT: Final[str]
BUFFER_RESOURCE_PERIODIC_SPILL_CHECK: Final[str]
BUFFER_RESOURCE_NUM_STREAMS: Final[str]

STREAMING_NUM_STREAMING_THREADS: Final[str]
STREAMING_MEMORY_RESERVE_TIMEOUT: Final[str]
STREAMING_ALLOW_OVERBOOKING_BY_DEFAULT: Final[str]

COMMUNICATOR_LOG: Final[str]

UCXX_PROGRESS_MODE: Final[str]

DEFAULTS: Final[Mapping[str, str]]

__all__: list[str]
