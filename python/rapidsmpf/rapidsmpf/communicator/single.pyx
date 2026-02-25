# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport make_shared, shared_ptr

from rapidsmpf._detail.exception_handling cimport ex_handler
from rapidsmpf.communicator.communicator cimport Communicator
from rapidsmpf.config cimport Options, cpp_Options
from rapidsmpf.progress_thread cimport ProgressThread, cpp_ProgressThread
from rapidsmpf.statistics cimport Statistics, cpp_Statistics


cdef extern from "<rapidsmpf/communicator/single.hpp>" nogil:
    cdef cppclass cpp_Single_Communicator "rapidsmpf::Single":
        cpp_Single_Communicator() except +ex_handler
        cpp_Single_Communicator(cpp_Options options) except +ex_handler
        cpp_Single_Communicator(
            cpp_Options options,
            shared_ptr[cpp_Statistics] statistics,
        ) except +ex_handler
        cpp_Single_Communicator(
            cpp_Options options,
            shared_ptr[cpp_ProgressThread] progress_thread,
        ) except +ex_handler


def new_communicator(Options options not None, progress=None):
    """
    Create a new RapidsMPF single-process communicator.

    Parameters
    ----------
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
        A new RapidsMPF single process communicator.
    """
    cdef Communicator ret = Communicator.__new__(Communicator)
    if progress is None:
        with nogil:
            ret._handle = make_shared[cpp_Single_Communicator](options._handle)
    elif isinstance(progress, ProgressThread):
        with nogil:
            ret._handle = make_shared[cpp_Single_Communicator](
                options._handle, (<ProgressThread>progress)._handle
            )
    elif isinstance(progress, Statistics):
        with nogil:
            ret._handle = make_shared[cpp_Single_Communicator](
                options._handle, (<Statistics>progress)._handle
            )
    else:
        raise TypeError(
            f"progress must be a ProgressThread, Statistics, or None, "
            f"got {type(progress).__name__}"
        )
    return ret
