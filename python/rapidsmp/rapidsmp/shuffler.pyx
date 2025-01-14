# Copyright (c) 2025, NVIDIA CORPORATION.

from libc.stdint cimport uint32_t
from libcpp.memory cimport make_unique
from rmm._cuda.stream cimport Stream


cdef class Shuffler:
    def __init__(
        self,
        Communicator comm,
        uint32_t total_num_partitions,
        stream,
        BufferResource br,
    ):
        self._stream = Stream(stream)
        self._comm = comm
        self._br = br
        self._handle = make_unique[cpp_Shuffler](
            comm._handle, total_num_partitions, self._stream.view(), br.ptr()
        )

    @property
    def comm(self):
        return self._comm
