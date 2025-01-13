# Copyright (c) 2025, NVIDIA CORPORATION.

from libcpp.memory cimport make_unique

cdef class Shuffler:
    def __init__(self, Communicator comm):
        # self._handle = make_unique[cpp_Shuffler](comm._handle)
        self._comm = comm

    @property
    def comm(self):
        return self._comm
