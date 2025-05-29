# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum
from typing import cast

class MemoryType(IntEnum):
    DEVICE = cast(int, ...)
    HOST = cast(int, ...)
