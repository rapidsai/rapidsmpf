# Copyright (c) 2025, NVIDIA CORPORATION.

from cython.operator cimport dereference as deref
from libcpp.memory cimport make_shared
from rapidsmp.communicator.communicator cimport Communicator


cdef class Statistics:
    """
    Track statistics across rapidsmp operations.

    Parameters
    ----------
    comm
        The communicator to use. If None, statistics is disabled.
    """
    def __cinit__(self, Communicator comm = None):
        if comm is None:
            self._handle = make_shared[cpp_Statistics]()
        else:
            self._handle = make_shared[cpp_Statistics](comm._handle)

    @property
    def enabled(self):
        return deref(self._handle).enabled()

    def report(self):
        return deref(self._handle).report().decode('UTF-8')
