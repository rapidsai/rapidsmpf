# Copyright (c) 2025, NVIDIA CORPORATION.

cdef extern from "<rapidsmp/buffer/buffer.hpp>" namespace "rapidsmp" nogil:
    cpdef enum class MemoryType(int):
        DEVICE
        HOST
