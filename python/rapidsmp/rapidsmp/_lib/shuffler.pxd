# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.exception_handler cimport libcudf_exception_handler


cdef extern from "rapidsmp/shuffler/shuffler.hpp" \
        namespace "rapidsmp::shuffler" nogil:

    cdef cppclass Shuffler:
        pass
