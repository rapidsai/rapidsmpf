# Copyright (c) 2025, NVIDIA CORPORATION.

cdef extern from "<rapidsmp/buffer/buffer.hpp>" nogil:
    cdef enum cpp_MemoryType:
        DEVICE
        HOST
