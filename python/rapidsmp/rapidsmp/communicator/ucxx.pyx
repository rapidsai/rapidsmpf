# Copyright (c) 2025, NVIDIA CORPORATION.

from cython.operator cimport dereference as deref
from libc.stdint cimport uint32_t
from libcpp.memory cimport (dynamic_pointer_cast, make_shared, nullptr,
                            shared_ptr, unique_ptr)
from libcpp.optional cimport nullopt
from libcpp.utility cimport move
from rapidsmp.communicator.communicator cimport *
from rapidsmp.communicator.ucxx cimport *
from ucxx._lib.libucxx cimport Address, UCXAddress, UCXWorker, Worker


cdef Communicator cpp_new_root_communicator(
    shared_ptr[Worker] worker,
    uint32_t nranks,
):
    cdef unique_ptr[cpp_UCXX_InitializedRank] ucxx_initialized_rank = init(
        worker,
        nranks,
        nullopt
    )
    cdef Communicator ret = Communicator.__new__(Communicator)
    ret._handle = make_shared[cpp_UCXX_Communicator](move(ucxx_initialized_rank))
    return ret


cdef Communicator cpp_new_communicator(
    shared_ptr[Worker] worker,
    uint32_t nranks,
    shared_ptr[Address] root_address,
):
    cdef unique_ptr[cpp_UCXX_InitializedRank] ucxx_initialized_rank = init(
        worker,
        nranks,
        root_address
    )
    cdef Communicator ret = Communicator.__new__(Communicator)
    ret._handle = make_shared[cpp_UCXX_Communicator](move(ucxx_initialized_rank))
    return ret


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
        # # Dereference twice: first the `get_if` result, then `shared_ptr`
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


def new_root_communicator(UCXWorker ucx_worker, uint32_t nranks):
    return cpp_new_root_communicator(ucx_worker._worker, nranks)


def new_communicator(
    UCXWorker ucx_worker,
    uint32_t nranks,
    UCXAddress root_ucxx_address
):
    return cpp_new_communicator(ucx_worker._worker, nranks, root_ucxx_address._address)


def new_root_communicator_no_worker(uint32_t nranks):
    return cpp_new_root_communicator(<shared_ptr[Worker]>nullptr, nranks)


def new_communicator_no_worker(uint32_t nranks, UCXAddress root_ucxx_address = None):
    return cpp_new_communicator(
        <shared_ptr[Worker]>nullptr,
        nranks,
        root_ucxx_address._address
    )
