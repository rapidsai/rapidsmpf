# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport make_shared, shared_ptr
from mpi4py cimport libmpi
from mpi4py.MPI cimport Intracomm

from rapidsmpf._detail.exception_handling cimport ex_handler
from rapidsmpf.communicator.communicator cimport Communicator, cpp_Logger
from rapidsmpf.config cimport Options
from rapidsmpf.progress_thread cimport ProgressThread, cpp_ProgressThread


cdef extern from "<rapidsmpf/communicator/mpi.hpp>" nogil:
    cdef cppclass cpp_MPI_Communicator "rapidsmpf::MPI":
        cpp_MPI_Communicator(
            libmpi.MPI_Comm comm,
            shared_ptr[cpp_ProgressThread],
            shared_ptr[cpp_Logger]
        ) except +ex_handler


def new_communicator(
    Intracomm comm not None,
    Options options not None,
    ProgressThread progress_thread not None,
):
    """
    Create a new RapidsMPF-MPI communicator based on an existing mpi4py communicator.

    Parameters
    ----------
    comm
        The existing mpi communicator from mpi4py.
    options
        Configuration options.
    progress_thread
        Progress thread for the communicator.

    Returns
    -------
    A new RapidsMPF-MPI communicator.
    """
    cdef Communicator ret = Communicator.__new__(Communicator)
    with nogil:
        ret._handle = make_shared[cpp_MPI_Communicator](
            comm.ob_mpi,
            progress_thread._handle,
            make_shared[cpp_Logger](options._handle)
        )
    return ret
