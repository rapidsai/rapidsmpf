# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport make_shared, shared_ptr

from rapidsmpf._detail.exception_handling cimport ex_handler
from rapidsmpf.communicator.communicator cimport Communicator
from rapidsmpf.config cimport Options, cpp_Options
from rapidsmpf.progress_thread cimport ProgressThread, cpp_ProgressThread


cdef extern from "<rapidsmpf/communicator/single.hpp>" nogil:
    cdef cppclass cpp_Single_Communicator "rapidsmpf::Single":
        cpp_Single_Communicator() except +ex_handler
        cpp_Single_Communicator(cpp_Options options) except +ex_handler
        cpp_Single_Communicator(
            cpp_Options options,
            shared_ptr[cpp_ProgressThread] progress_thread,
        ) except +ex_handler


def new_communicator(Options options not None, ProgressThread progress_thread = None):
    """
    Create a new RapidsMPF single-process communicator.

    Parameters
    ----------
    options
        Configuration options.
    progress_thread
        An existing progress thread to share. If None, a new one is created.

    Returns
    -------
        A new RapidsMPF single process communicator.
    """
    cdef Communicator ret = Communicator.__new__(Communicator)
    if progress_thread is not None:
        with nogil:
            ret._handle = make_shared[cpp_Single_Communicator](
                options._handle, progress_thread._handle
            )
    else:
        with nogil:
            ret._handle = make_shared[cpp_Single_Communicator](options._handle)
    return ret
