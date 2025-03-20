# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport make_shared
from mpi4py cimport libmpi
from mpi4py.MPI cimport Intracomm
from rapidsmp.communicator.communicator cimport Communicator


cdef extern from "<rapidsmp/communicator/mpi.hpp>" nogil:
    cdef cppclass cpp_MPI_Communicator "rapidsmp::MPI":
        cpp_MPI_Communicator() except +
        cpp_MPI_Communicator(libmpi.MPI_Comm comm) except +


cpdef Communicator new_communicator(Intracomm comm):
    """
    Create a new rapidsmp-mpi communicator based on an existing mpi4py communicator.

    Parameters
    ----------
    comm
        The existing mpi communicator from mpi4py.

    Returns
    -------
        A new rapidsmp-mpi communicator.
    """
    cdef Communicator ret = Communicator.__new__(Communicator)
    ret._handle = make_shared[cpp_MPI_Communicator](comm.ob_mpi)
    return ret
