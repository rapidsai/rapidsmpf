# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from rapidsmpf.memory.buffer import MemoryType


cdef content_description_from_cpp(cpp_ContentDescription cd):
    cdef dict content_sizes = {}
    for mem_type in MemoryType:
        content_sizes[mem_type] = cd.content_size(mem_type)
    return ContentDescription(content_sizes, cd.spillable())


@dataclass
class ContentDescription:
    content_sizes: dict[MemoryType, int]
    spillable: bool
