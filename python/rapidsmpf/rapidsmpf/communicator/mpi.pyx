# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport make_shared
from mpi4py cimport libmpi
from mpi4py.MPI cimport Intracomm

from rapidsmpf._detail.exception_handling cimport ex_handler
from rapidsmpf.communicator.communicator cimport Communicator
from rapidsmpf.config cimport Options, cpp_Options


cdef extern from "<rapidsmpf/communicator/mpi.hpp>" nogil:
    cdef cppclass cpp_MPI_Communicator "rapidsmpf::MPI":
        cpp_MPI_Communicator() except +ex_handler
        cpp_MPI_Communicator(
            libmpi.MPI_Comm comm, cpp_Options options
        ) except +ex_handler


def new_communicator(Intracomm comm not None, Options options not None):
    """
    Create a new RapidsMPF-MPI communicator based on an existing mpi4py communicator.

    Parameters
    ----------
    comm
        The existing mpi communicator from mpi4py.
    options
        Configuration options.

    Returns
    -------
        A new RapidsMPF-MPI communicator.
    """
    cdef Communicator ret = Communicator.__new__(Communicator)
    with nogil:
        ret._handle = make_shared[cpp_MPI_Communicator](comm.ob_mpi, options._handle)
    return ret
