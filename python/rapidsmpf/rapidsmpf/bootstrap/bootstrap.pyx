# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport dynamic_pointer_cast, shared_ptr

from rapidsmpf.communicator.communicator cimport Communicator, cpp_Communicator
from rapidsmpf.config cimport Options, cpp_Options


cdef extern from "<rapidsmpf/bootstrap/bootstrap.hpp>" namespace \
  "rapidsmpf::bootstrap" nogil:
    cpdef enum class Backend(int):
        AUTO
        FILE


cdef extern from "<rapidsmpf/communicator/ucxx.hpp>" namespace \
  "rapidsmpf::ucxx" nogil:
    cdef cppclass cpp_UCXX_Communicator "rapidsmpf::ucxx::UCXX":
        pass


cdef extern from "<rapidsmpf/bootstrap/ucxx.hpp>" namespace \
  "rapidsmpf::bootstrap" nogil:
    shared_ptr[cpp_UCXX_Communicator] create_ucxx_comm(
        Backend backend,
        cpp_Options options,
    ) except +


def create_ucxx_comm(Backend backend = Backend.AUTO, options = None):
    """
    Create a UCXX communicator using the bootstrap backend.

    This function bootstraps a UCXX-based communicator using the selected
    coordination backend (currently file-based), relying on environment
    variables such as ``RAPIDSMPF_RANK``, ``RAPIDSMPF_NRANKS``, and
    ``RAPIDSMPF_COORD_DIR``.

    Parameters
    ----------
    backend
        Backend to use for coordination. By default, ``Backend.AUTO`` is used,
        which currently resolves to the file-based backend.
    options
        Configuration options for the UCXX communicator. If ``None``, a default
        `rapidsmpf.config.Options` instance is used.

    Returns
    -------
    rapidsmpf.communicator.communicator.Communicator
        A new RapidsMPF-UCXX communicator instance.
    """
    cdef Communicator ret = Communicator.__new__(Communicator)
    cdef shared_ptr[cpp_UCXX_Communicator] ucxx_comm
    cdef shared_ptr[cpp_Communicator] base_comm
    cdef Options cpp_options

    if options is None:
        cpp_options = Options()
    else:
        if not isinstance(options, Options):
            raise TypeError(
                "options must be a rapidsmpf.config.Options instance or None"
            )
        cpp_options = <Options>options

    with nogil:
        ucxx_comm = create_ucxx_comm(backend, cpp_options._handle)
        base_comm = dynamic_pointer_cast[cpp_Communicator, cpp_UCXX_Communicator](
            ucxx_comm
        )
        ret._handle = base_comm

    return ret
