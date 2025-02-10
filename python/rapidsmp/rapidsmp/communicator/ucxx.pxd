# Copyright (c) 2025, NVIDIA CORPORATION.

from libc.stdint cimport uint16_t, uint32_t
from libcpp.memory cimport make_shared, shared_ptr, unique_ptr
from libcpp.optional cimport nullopt, nullopt_t, optional
from libcpp.pair cimport pair
from libcpp.string cimport string
from rapidsmp.communicator.communicator cimport Communicator
from ucxx._lib.libucxx cimport Address, UCXAddress, UCXWorker, Worker


cdef extern from "<variant>" namespace "std" nogil:
    # cdef cppclass variant[T1, T2]:
    #     variant() except +
    #     T1& get[T1]() except +
    #     T2& get[T2]() except +
    #     size_t index()

    cdef cppclass variant:
        variant& operator=(variant&)
        size_t index()

    cdef T get[T](...)
    cdef T* get_if[T](...)


# cdef extern from "<ucxx/api.h>" nogil:
#     cdef cppclass cpp_UCXX_Address "ucxx::Address":
#         string getString() except +
#
#     cdef cppclass cpp_UCXX_Worker "ucxx::Worker":
#         pass


# cdef class UCXAddress:
#     cdef shared_ptr[cpp_UCXX_Address] _address


# cdef class UCXWorker:
#     cdef shared_ptr[cpp_UCXX_Worker] _worker
#
#     # cdef shared_ptr[cpp_UCXX_Worker] get_ucxx_shared_ptr(self)


cdef extern from "<rapidsmp/communicator/ucxx.hpp>" namespace "rapidsmp::ucxx" nogil:
    ctypedef int Rank

    ctypedef pair[string, uint16_t] HostPortPair

    # ctypedef variant[HostPortPair, shared_ptr[cpp_UCXX_Address]] RemoteAddress
    ctypedef variant RemoteAddress

    cdef cppclass cpp_UCXX_ListenerAddress "rapidsmp::ucxx::ListenerAddress":
        RemoteAddress address
        Rank rank

    cdef cppclass cpp_UCXX_InitializedRank "rapidsmp::ucxx::InitializedRank":
        pass

    unique_ptr[cpp_UCXX_InitializedRank] init(
        # shared_ptr[cpp_UCXX_Worker] worker,
        shared_ptr[Worker] worker,
        uint32_t nranks,
        # optional[RemoteAddress] remote_address
        # shared_ptr[cpp_UCXX_Address] remote_address
        shared_ptr[Address] remote_address
    )
    unique_ptr[cpp_UCXX_InitializedRank] init(
        # shared_ptr[cpp_UCXX_Worker] worker,
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
