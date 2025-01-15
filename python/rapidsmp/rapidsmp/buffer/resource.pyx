# Copyright (c) 2025, NVIDIA CORPORATION.

from libcpp.memory cimport make_shared
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource


cdef class BufferResource:
    """
    A buffer resource.

    Parameters
    ----------
    device_mr
        Reference to the RMM device memory resource used for device allocations.
    """

    def __cinit__(self, DeviceMemoryResource device_mr):
        self._handle = make_shared[cpp_BufferResource](device_mr.get_mr())

    cdef cpp_BufferResource* ptr(self):
        """
        A raw pointer to the underlying C++ `BufferResource`.

        Returns
        -------
            The raw pointer.
        """
        return self._handle.get()
