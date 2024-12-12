# Copyright (c) 2024, NVIDIA CORPORATION.

cdef extern from "<rapidsmp/communicator/communicator.hpp>" nogil:

    cdef cppclass cpp_Communicator "rapidsmp::Communicator":
        pass