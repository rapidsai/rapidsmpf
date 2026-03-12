# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

cdef extern from "<rapidsmpf/memory/buffer.hpp>" namespace "rapidsmpf" nogil:
    cpdef enum class MemoryType(int):
        DEVICE
        PINNED_HOST
        HOST
