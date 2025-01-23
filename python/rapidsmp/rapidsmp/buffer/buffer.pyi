# Copyright (c) 2025, NVIDIA CORPORATION.

from enum import IntEnum
from typing import cast

class MemoryType(IntEnum):
    DEVICE = cast(int, ...)
    HOST = cast(int, ...)
