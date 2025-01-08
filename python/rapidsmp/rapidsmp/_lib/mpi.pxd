# Copyright (c) 2025-2025, NVIDIA CORPORATION.

cdef extern from "<rapidsmp/communicator/mpi.hpp>" nogil:

    cdef cppclass cpp_MPI_Communicator "rapidsmp::MPI":
        pass
