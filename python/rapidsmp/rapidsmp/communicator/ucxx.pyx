# Copyright (c) 2025, NVIDIA CORPORATION.

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

    cdef cppclass cpp_UCXX_ListenerAddress "rapidsmp::ucxx::ListenerAddress":
        RemoteAddress address
        Rank rank

    cdef cppclass cpp_UCXX_InitializedRank "rapidsmp::ucxx::InitializedRank":
        pass

    unique_ptr[cpp_UCXX_InitializedRank] init(
        shared_ptr[Worker] worker,
        uint32_t nranks,
        shared_ptr[Address] remote_address
    )

    unique_ptr[cpp_UCXX_InitializedRank] init(
        shared_ptr[Worker] worker,
        uint32_t nranks,
        nullopt_t remote_address
    )

    cdef cppclass cpp_UCXX_Communicator "rapidsmp::ucxx::UCXX":
        cpp_UCXX_Communicator() except +
        cpp_UCXX_Communicator(
            unique_ptr[cpp_UCXX_InitializedRank] ucxx_initialized_rank
        ) except +
        cpp_UCXX_ListenerAddress listener_address()
        void barrier() except +


cdef Communicator cpp_new_communicator(
    shared_ptr[Worker] worker,
    uint32_t nranks,
    shared_ptr[Address] root_address,
):
    cdef unique_ptr[cpp_UCXX_InitializedRank] ucxx_initialized_rank

    if root_address == <shared_ptr[Address]>nullptr:
        ucxx_initialized_rank = init(worker, nranks, nullopt)
    else:
        ucxx_initialized_rank = init(worker, nranks, root_address)
    cdef Communicator ret = Communicator.__new__(Communicator)
    ret._handle = make_shared[cpp_UCXX_Communicator](move(ucxx_initialized_rank))
    return ret


def new_communicator(
    uint32_t nranks = 1,
    UCXWorker ucx_worker = None,
    UCXAddress root_ucxx_address = None
):
    if ucx_worker is None:
        ucx_worker_ptr = <shared_ptr[Worker]>nullptr
    else:
        ucx_worker_ptr = ucx_worker.get_ucxx_shared_ptr()
    if root_ucxx_address is None:
        root_ucxx_address_ptr = <shared_ptr[Address]>nullptr
    else:
        root_ucxx_address_ptr = root_ucxx_address.get_ucxx_shared_ptr()

    return cpp_new_communicator(ucx_worker_ptr, nranks, root_ucxx_address_ptr)


def get_root_ucxx_address(Communicator comm):
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
    cdef shared_ptr[cpp_UCXX_Communicator] ucxx_comm = (
        dynamic_pointer_cast[cpp_UCXX_Communicator, cpp_Communicator](
            comm._handle
        )
    )

    deref(ucxx_comm).barrier()
