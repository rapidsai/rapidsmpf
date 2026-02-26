# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport dynamic_pointer_cast, shared_ptr

from rapidsmpf._detail.exception_handling cimport ex_handler
from rapidsmpf.communicator.communicator cimport Communicator, cpp_Communicator
from rapidsmpf.config cimport Options, cpp_Options


cdef extern from "<rapidsmpf/bootstrap/backend.hpp>" namespace \
  "rapidsmpf::bootstrap" nogil:
    cpdef enum class BackendType(int):
        AUTO
        FILE


cdef extern from "<rapidsmpf/communicator/ucxx.hpp>" namespace \
  "rapidsmpf::ucxx" nogil:
    cdef cppclass cpp_UCXX_Communicator "rapidsmpf::ucxx::UCXX":
        pass


cdef extern from "<rapidsmpf/bootstrap/utils.hpp>" nogil:
    bint cpp_is_running_with_rrun \
        "rapidsmpf::bootstrap::is_running_with_rrun"() except +ex_handler

    bint cpp_is_running_with_slurm \
        "rapidsmpf::bootstrap::is_running_with_slurm"() except +ex_handler

    int cpp_get_rank \
        "rapidsmpf::bootstrap::get_rank"() except +ex_handler

    int cpp_get_nranks \
        "rapidsmpf::bootstrap::get_nranks"() except +ex_handler

cdef extern from "<rapidsmpf/bootstrap/ucxx.hpp>" nogil:
    shared_ptr[cpp_UCXX_Communicator] cpp_create_ucxx_comm \
        "rapidsmpf::bootstrap::create_ucxx_comm"(
            BackendType type,
            cpp_Options options,
        ) except +ex_handler


def create_ucxx_comm(BackendType type = BackendType.AUTO, options = None):
    """
    Create a UCXX communicator using the bootstrap backend.

    This function bootstraps a UCXX-based communicator using the selected
    coordination backend (currently file-based), relying on environment
    variables such as ``RAPIDSMPF_RANK``, ``RAPIDSMPF_NRANKS``, and
    ``RAPIDSMPF_COORD_DIR``.

    Parameters
    ----------
    type
        Backend type to use for coordination. By default, ``BackendType.AUTO`` is used,
        which currently resolves to the file-based backend.
    options
        Configuration options for the UCXX communicator. If ``None``, a default
        `rapidsmpf.config.Options` instance is used.

    Returns
    -------
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
        ucxx_comm = cpp_create_ucxx_comm(type, cpp_options._handle)
        base_comm = dynamic_pointer_cast[cpp_Communicator, cpp_UCXX_Communicator](
            ucxx_comm
        )
        ret._handle = base_comm

    return ret


def is_running_with_rrun():
    """
    Check whether the current process was launched via ``rrun``.

    This helper inspects the bootstrap environment (e.g. the presence of
    ``RAPIDSMPF_RANK``) to determine if the process is running under
    ``rrun``-managed bootstrap mode.

    Returns
    -------
    ``True`` if running under ``rrun`` bootstrap mode, ``False`` otherwise.
    """
    cdef bint ret
    with nogil:
        ret = cpp_is_running_with_rrun()
    return bool(ret)


def is_running_with_slurm():
    """
    Check whether the current process is running under Slurm with PMIx.

    This helper detects Slurm environment by checking for PMIx namespace
    or Slurm job step environment variables.

    Returns
    -------
    ``True`` if running under Slurm with PMIx, ``False`` otherwise.
    """
    cdef bint ret
    with nogil:
        ret = cpp_is_running_with_slurm()
    return bool(ret)


def get_rank():
    """
    Get the current bootstrap rank.

    This helper retrieves the rank of the current process when running with a
    bootstrap launcher (rrun or Slurm). Checks environment variables in order:
    1. RAPIDSMPF_RANK (set by rrun)
    2. PMIX_RANK (set by PMIx)
    3. SLURM_PROCID (set by Slurm)

    Returns
    -------
    Rank of the current process.

    Raises
    ------
    RuntimeError
        If not running with a bootstrap launcher or if the environment
        variable cannot be parsed.
    """
    cdef int ret
    with nogil:
        ret = cpp_get_rank()
    return ret


def get_nranks():
    """
    Get the number of ``rrun`` ranks.

    This helper retrieves the number of ranks when running with ``rrun``.
    The number of ranks is fetched from the ``RAPIDSMPF_NRANKS`` environment variable.

    Returns
    -------
    int
        Number of ranks.

    Raises
    ------
    RuntimeError
        If not running with ``rrun`` or if ``RAPIDSMPF_NRANKS`` is not set
        or cannot be parsed.
    """
    cdef int ret
    with nogil:
        ret = cpp_get_nranks()
    return ret
