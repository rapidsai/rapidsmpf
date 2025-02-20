# Copyright (c) 2025, NVIDIA CORPORATION.

from cython.operator cimport dereference as deref
from libcpp.memory cimport make_shared


cdef class Statistics:
    def __cinit__(self, int nranks):
        self._handle = make_shared[cpp_Statistics](nranks)

    @property
    def enabled(self):
        return deref(self._handle).enabled()

    def report(self):
        return deref(self._handle).report().decode('UTF-8')
