# Copyright (c) 2024, NVIDIA CORPORATION.


cdef extern from "rapidsmp/shuffler/shuffler.hpp" \
        namespace "rapidsmp::shuffler" nogil:

    cdef cppclass Shuffler:
        pass
