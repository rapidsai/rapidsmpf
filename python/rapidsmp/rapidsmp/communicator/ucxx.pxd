# Copyright (c) 2025, NVIDIA CORPORATION.

from libc.stdint cimport uint16_t, uint32_t
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.optional cimport nullopt_t
from libcpp.pair cimport pair
from libcpp.string cimport string
from ucxx._lib.libucxx cimport Address, Worker


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
