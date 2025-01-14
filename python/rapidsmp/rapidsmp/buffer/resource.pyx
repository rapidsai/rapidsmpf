# Copyright (c) 2025, NVIDIA CORPORATION.

from libcpp.memory cimport make_shared
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource


cdef class BufferResource:
    def __cinit__(self, DeviceMemoryResource device_mr):
        self._handle = make_shared[cpp_BufferResource](device_mr.get_mr())

    cdef cpp_BufferResource* ptr(self):
        return self._handle.get()
