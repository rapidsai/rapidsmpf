# Copyright (c) 2025, NVIDIA CORPORATION.

from cython.operator cimport dereference as deref


cdef class Communicator:

    @property
    def rank(self):
        return deref(self._handle).rank()
