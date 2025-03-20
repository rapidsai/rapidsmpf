# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

cdef extern from "<rapidsmp/buffer/buffer.hpp>" namespace "rapidsmp" nogil:
    cpdef enum class MemoryType(int):
        DEVICE
        HOST
