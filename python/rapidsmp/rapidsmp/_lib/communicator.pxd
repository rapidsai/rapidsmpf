# Copyright (c) 2025, NVIDIA CORPORATION.

cdef extern from "<rapidsmp/communicator/communicator.hpp>" nogil:

    cdef cppclass cpp_Communicator "rapidsmp::Communicator":
        pass
