# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport make_shared, shared_ptr
from mpi4py cimport libmpi
from mpi4py.MPI cimport Intracomm

from rapidsmpf._detail.exception_handling cimport ex_handler
from rapidsmpf.communicator.communicator cimport Communicator
from rapidsmpf.config cimport Options, cpp_Options
from rapidsmpf.progress_thread cimport ProgressThread, cpp_ProgressThread
from rapidsmpf.statistics cimport Statistics, cpp_Statistics


cdef extern from "<rapidsmpf/communicator/mpi.hpp>" nogil:
    cdef cppclass cpp_MPI_Communicator "rapidsmpf::MPI":
        cpp_MPI_Communicator() except +ex_handler
        cpp_MPI_Communicator(
            libmpi.MPI_Comm comm, cpp_Options options
        ) except +ex_handler
        cpp_MPI_Communicator(
            libmpi.MPI_Comm comm,
            cpp_Options options,
            shared_ptr[cpp_Statistics] statistics,
        ) except +ex_handler
        cpp_MPI_Communicator(
            libmpi.MPI_Comm comm,
            cpp_Options options,
            shared_ptr[cpp_ProgressThread] progress_thread,
        ) except +ex_handler


def new_communicator(
    Intracomm comm not None,
    Options options not None,
    progress=None,
):
    """
    Create a new RapidsMPF-MPI communicator based on an existing mpi4py communicator.

    Parameters
    ----------
    comm
        The existing mpi communicator from mpi4py.
    options
        Configuration options.
    progress
        Either a :class:`~rapidsmpf.statistics.Statistics` instance (to
        create a new progress thread with those statistics) or a
        :class:`~rapidsmpf.progress_thread.ProgressThread` instance (to
        share an existing progress thread). If ``None``, a new progress
        thread with disabled statistics is created.

    Returns
    -------
        A new RapidsMPF-MPI communicator.
    """
    cdef Communicator ret = Communicator.__new__(Communicator)
    if progress is None:
        with nogil:
            ret._handle = make_shared[cpp_MPI_Communicator](
                comm.ob_mpi, options._handle
            )
    elif isinstance(progress, ProgressThread):
        with nogil:
            ret._handle = make_shared[cpp_MPI_Communicator](
                comm.ob_mpi, options._handle, (<ProgressThread>progress)._handle
            )
    elif isinstance(progress, Statistics):
        with nogil:
            ret._handle = make_shared[cpp_MPI_Communicator](
                comm.ob_mpi, options._handle, (<Statistics>progress)._handle
            )
    else:
        raise TypeError(
            f"progress must be a ProgressThread, Statistics, or None, "
            f"got {type(progress).__name__}"
        )
    return ret
