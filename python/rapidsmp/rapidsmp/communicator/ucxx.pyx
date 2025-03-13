# Copyright (c) 2025, NVIDIA CORPORATION.

from enum import Enum

from cython.operator cimport dereference as deref
from libc.stdint cimport uint16_t, uint32_t
from libcpp.memory cimport (dynamic_pointer_cast, make_shared, nullptr,
                            shared_ptr, unique_ptr)
from libcpp.optional cimport nullopt, nullopt_t
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.utility cimport move
from rapidsmp.communicator.communicator cimport *
from rapidsmp.communicator.ucxx cimport *
from ucxx._lib.libucxx cimport Address, UCXAddress, UCXWorker, Worker


cdef extern from "<variant>" namespace "std" nogil:
    cdef cppclass variant:
        variant& operator=(variant&)
        size_t index()

    cdef T get[T](...)
    cdef T* get_if[T](...)


cdef extern from "<rapidsmp/communicator/ucxx.hpp>" namespace "rapidsmp::ucxx" nogil:
    ctypedef int Rank

    ctypedef pair[string, uint16_t] HostPortPair

    ctypedef variant RemoteAddress

    cpdef enum class ProgressMode(int):
        Blocking
        Polling
        ThreadBlocking
        ThreadPolling

    cdef cppclass cpp_UCXX_ListenerAddress "rapidsmp::ucxx::ListenerAddress":
        RemoteAddress address
        Rank rank

    cdef cppclass cpp_UCXX_InitializedRank "rapidsmp::ucxx::InitializedRank":
        pass

    unique_ptr[cpp_UCXX_InitializedRank] init(
        shared_ptr[Worker] worker,
        uint32_t nranks,
        shared_ptr[Address] remote_address,
        ProgressMode progress_mode
    )

    unique_ptr[cpp_UCXX_InitializedRank] init(
        shared_ptr[Worker] worker,
        uint32_t nranks,
        nullopt_t remote_address,
        ProgressMode progress_mode
    )

    cdef cppclass cpp_UCXX_Communicator "rapidsmp::ucxx::UCXX":
        cpp_UCXX_Communicator() except +
        cpp_UCXX_Communicator(
            unique_ptr[cpp_UCXX_InitializedRank] ucxx_initialized_rank
        ) except +
        cpp_UCXX_ListenerAddress listener_address()
        void barrier() except +


cdef Communicator cpp_new_communicator(
    uint32_t nranks,
    shared_ptr[Worker] worker,
    shared_ptr[Address] root_address,
    ProgressMode progress_mode,
):
    cdef unique_ptr[cpp_UCXX_InitializedRank] ucxx_initialized_rank

    if root_address == <shared_ptr[Address]>nullptr:
        ucxx_initialized_rank = init(worker, nranks, nullopt, progress_mode)
    else:
        ucxx_initialized_rank = init(worker, nranks, root_address, progress_mode)
    cdef Communicator ret = Communicator.__new__(Communicator)
    ret._handle = make_shared[cpp_UCXX_Communicator](move(ucxx_initialized_rank))
    return ret


def new_communicator(
    uint32_t nranks,
    UCXWorker ucx_worker,
    UCXAddress root_ucxx_address,
    ProgressMode progress_mode = ProgressMode.ThreadBlocking,
):
    """
    Create a new UCXX communicator with the given number of ranks.

    An existing UCXWorker may be specified, otherwise one will be created. The root rank
    is created if no `root_ucxx_address` is specific, all other ranks must specify the
    the address of the root rank via that argument.

    Parameters
    ----------
    nranks
        The number of ranks in the cluster.
    ucx_worker
        An existing UCXX worker to use if specified, otherwise one will be created.
    root_ucxx_address
        The UCXX address of the root rank (only specified for non-root ranks).
    progress_mode
        The progress mode to use with the UCXX worker.

    Returns
    -------
        A new rapidsmp-ucxx communicator.
    """
    if ucx_worker is None:
        ucx_worker_ptr = <shared_ptr[Worker]>nullptr
    else:
        ucx_worker_ptr = ucx_worker.get_ucxx_shared_ptr()
    if root_ucxx_address is None:
        root_ucxx_address_ptr = <shared_ptr[Address]>nullptr
    else:
        root_ucxx_address_ptr = root_ucxx_address.get_ucxx_shared_ptr()

    return cpp_new_communicator(
        nranks,
        ucx_worker_ptr,
        root_ucxx_address_ptr,
        progress_mode
    )


def get_root_ucxx_address(Communicator comm):
    """
    Get the address of the communicator's UCXX worker.

    This function is intended to be called from the root rank to communicate
    to other processes how to reach the root, but it will return the address of
    UCXX worker of other ranks too.

    Parameters
    ----------
    comm
        The rapidsmp-ucxx communicator.

    Raises
    ------
    NotImplementedError
        If the communicator was created with a HostPortPair, which is not yet
        supported.

    Returns
    -------
        A string with the UCXX worker address.
    """
    cdef shared_ptr[cpp_UCXX_Communicator] ucxx_comm = (
        dynamic_pointer_cast[cpp_UCXX_Communicator, cpp_Communicator](
            comm._handle
        )
    )
    cdef cpp_UCXX_ListenerAddress listener_address = deref(ucxx_comm).listener_address()

    cdef shared_ptr[Address]* address
    cdef HostPortPair* host_port_pair

    if address := get_if[shared_ptr[Address]](&listener_address.address):
        # Dereference twice: first the `get_if` result, then `shared_ptr`
        return deref(deref(address)).getString()
    elif host_port_pair := get_if[HostPortPair](&listener_address.address):
        raise NotImplementedError("Accepting HostPortPair is not implemented yet")
        assert host_port_pair  # Prevent "defined but unused" error


def barrier(Communicator comm):
    """
    Execute a barrier on the UCXX communicator.

    Ensures all ranks connected to the root and all ranks reached the barrier before
    continuing.

    Notes
    -----
    Executing this barrier is required after the ranks are bootstrapped to ensure
    everyone is connected to the root. An alternative barrier, such as
    `MPI_Barrier` will not suffice for that purpose.
    """
    cdef shared_ptr[cpp_UCXX_Communicator] ucxx_comm = (
        dynamic_pointer_cast[cpp_UCXX_Communicator, cpp_Communicator](
            comm._handle
        )
    )
    deref(ucxx_comm).barrier()
