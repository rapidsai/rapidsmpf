# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import functools
from typing import cast

import distributed
import packaging.version


@functools.lru_cache
def distributed_version() -> str:
    return cast(str, distributed.__version__)


@functools.lru_cache
def DISTRIBUTED_2025_4_0() -> bool:
    return cast(
        bool,
        packaging.version.parse(distributed_version())
        >= packaging.version.parse("2025.4.0"),
    )
