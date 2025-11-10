# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from rapidsmpf.buffer.buffer import MemoryType

@dataclass
class ContentDescription:
    content_sizes: dict[MemoryType, int]
    spillable: bool
