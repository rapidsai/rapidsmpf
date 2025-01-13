# Copyright (c) 2025, NVIDIA CORPORATION.

from mpi4py cimport MPI
from mpi4py cimport libmpi
from mpi4py.MPI cimport Intracomm

from libcpp.memory cimport make_shared, shared_ptr

from rapidsmp.communicator.communicator cimport Communicator

cdef extern from "<rapidsmp/communicator/mpi.hpp>" nogil:
    cdef cppclass cpp_MPI_Communicator "rapidsmp::MPI":
        cpp_MPI_Communicator() except +
        cpp_MPI_Communicator(libmpi.MPI_Comm comm) except +


cpdef Communicator new_communicator(Intracomm comm):
    cdef Communicator ret = Communicator()
    ret._handle = make_shared[cpp_MPI_Communicator](comm.ob_mpi)
    return ret
