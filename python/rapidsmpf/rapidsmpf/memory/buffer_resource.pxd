# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stddef cimport size_t
from libc.stdint cimport int64_t, uint64_t
from libcpp cimport bool as bool_t
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.optional cimport optional
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from rmm.librmm.cuda_stream_pool cimport cuda_stream_pool
from rmm.librmm.cuda_stream_ref cimport stream_ref
from rmm.librmm.memory_resource cimport (any_resource, device_accessible,
                                         device_async_resource_ref)
from rmm.pylibrmm.cuda_stream_pool cimport CudaStreamPool
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from rapidsmpf._detail.exception_handling cimport ex_handler
from rapidsmpf.config cimport Options, cpp_Options
from rapidsmpf.memory.buffer cimport Buffer, MemoryType, cpp_Buffer
from rapidsmpf.memory.host_memory_resource cimport (HostMemoryResource,
                                                    cpp_HostMemoryResource)
from rapidsmpf.memory.memory_reservation cimport cpp_MemoryReservation
from rapidsmpf.memory.pinned_memory_resource cimport (PinnedMemoryResource,
                                                      cpp_PinnedMemoryResource,
                                                      cpp_PinnedPoolProperties)
from rapidsmpf.memory.reservation_attributes cimport cpp_ReservationAttributes
from rapidsmpf.memory.spill_manager cimport SpillManager, cpp_SpillManager
from rapidsmpf.rmm_resource_adaptor cimport (RmmResourceAdaptor,
                                             cpp_RmmResourceAdaptor)
from rapidsmpf.statistics cimport Statistics, cpp_Statistics
from rapidsmpf.utils.time cimport cpp_Duration


cdef extern from "<filesystem>" nogil:
    cdef cppclass cpp_path "std::filesystem::path":
        cpp_path() noexcept
        cpp_path(string) except +
        string string() except +

cdef extern from "<rapidsmpf/disk/disk_resource.hpp>" nogil:
    cdef cppclass cpp_DiskResource "rapidsmpf::DiskResource":
        const cpp_path& directory() noexcept

cdef extern from "<rapidsmpf/memory/reservation_events.hpp>" namespace "rapidsmpf":
    cpdef enum class ReservationRequestPolicy(unsigned int):
        IMMEDIATE
        OVERBOOK
        SPILL
        WAIT
        WAIT_OR_OVERBOOK
        WAIT_OR_FAIL
    cpdef enum class ReservationRequestOutcome(unsigned int):
        GRANTED
        EMPTY
        FAILED
        CANCELLED
    cpdef enum class ReservationResizeReason(unsigned int):
        CONSUMED
        SPLIT_OFF
    cpdef enum class ReservationReleaseReason(unsigned int):
        CLEARED
        RELEASED_CONSUMED
        RELEASED_SPLIT

    cdef cppclass cpp_RequestSubmittedEvent "rapidsmpf::RequestSubmittedEvent":
        uint64_t id
        size_t size
        vector[MemoryType] memory_types
        ReservationRequestPolicy policy
        optional[int64_t] net_memory_delta
        cpp_ReservationAttributes attributes

    cdef cppclass cpp_RequestWaitingEvent "rapidsmpf::RequestWaitingEvent":
        uint64_t id

    cdef cppclass cpp_RequestResolvedEvent "rapidsmpf::RequestResolvedEvent":
        uint64_t id
        ReservationRequestOutcome outcome
        uint64_t reservation_id
        optional[MemoryType] memory_type
        size_t size
        size_t overbooked_bytes
        bool_t timed_out
        string error

    cdef cppclass cpp_ReservationCreatedEvent "rapidsmpf::ReservationCreatedEvent":
        uint64_t id
        MemoryType memory_type
        size_t size
        cpp_ReservationAttributes attributes
        uint64_t request_id
        uint64_t parent_id

    cdef cppclass cpp_ReservationResizedEvent "rapidsmpf::ReservationResizedEvent":
        uint64_t id
        size_t size
        ReservationResizeReason reason

    cdef cppclass cpp_ReservationReleasedEvent "rapidsmpf::ReservationReleasedEvent":
        uint64_t id
        ReservationReleaseReason reason

    cdef cppclass cpp_MemoryLimitChangedEvent "rapidsmpf::MemoryLimitChangedEvent":
        MemoryType memory_type
        int64_t configured_limit

    cdef cppclass cpp_ReservationEventData "rapidsmpf::ReservationEventData":
        size_t index() noexcept

    cdef cppclass cpp_ReservationEvent "rapidsmpf::ReservationEvent":
        uint64_t sequence
        int64_t steady_time_ns
        cpp_ReservationEventData data

    ctypedef vector[cpp_ReservationEvent] cpp_ReservationEventBatch \
        "rapidsmpf::ReservationEventBatch"

cdef extern from "<rapidsmpf/memory/buffer_resource.hpp>" nogil:
    cdef enum class AllowOverbooking"rapidsmpf::AllowOverbooking"(bool_t):
        NO
        YES

cdef extern from "<rapidsmpf/memory/buffer_resource.hpp>" nogil:
    cdef cppclass cpp_StreamPool "rapidsmpf::StreamPool":
        cpp_StreamPool(shared_ptr[cuda_stream_pool])

    cdef cppclass cpp_BufferResource "rapidsmpf::BufferResource":
        @staticmethod
        shared_ptr[cpp_BufferResource] create(
            any_resource[device_accessible],
            optional[cpp_PinnedPoolProperties],
            unordered_map[MemoryType, int64_t],
            optional[cpp_Duration],
            shared_ptr[cpp_StreamPool],
            shared_ptr[cpp_Statistics],
            optional[cpp_path],
            cpp_Duration,
            AllowOverbooking,
        ) except +ex_handler
        int64_t memory_available_for_reservation(
            MemoryType mem_type
        ) except +ex_handler
        int64_t memory_available(MemoryType mem_type) except +ex_handler
        int64_t memory_limit(MemoryType mem_type) noexcept
        bool_t has_memory_resource(MemoryType mem_type) noexcept
        void set_memory_limit(MemoryType mem_type, int64_t limit) except +ex_handler
        void start_reservation_event_recording() except +ex_handler
        cpp_ReservationEventBatch drain_reservation_events() except +ex_handler
        cpp_ReservationEventBatch stop_reservation_event_recording() except +ex_handler
        cpp_SpillManager &spill_manager() except +ex_handler
        size_t release(cpp_MemoryReservation&, size_t) except +ex_handler
        shared_ptr[cpp_Statistics] statistics() except +ex_handler
        device_async_resource_ref device_mr() noexcept
        cpp_RmmResourceAdaptor& device_mr_adaptor() noexcept
        optional[cpp_PinnedMemoryResource] try_pinned_mr() except +ex_handler
        const shared_ptr[cpp_DiskResource]& disk_resource() noexcept
        cpp_Duration memory_reserve_timeout() noexcept
        AllowOverbooking allow_overbooking_by_default() noexcept
        unique_ptr[cpp_Buffer] make_buffer(
            size_t size,
            stream_ref stream,
            cpp_MemoryReservation& reservation,
        ) except +ex_handler

cdef class BufferResource:
    cdef object __weakref__
    cdef shared_ptr[cpp_BufferResource] _handle
    cdef readonly SpillManager spill_manager
    cdef cpp_BufferResource* ptr(self)
    cdef DeviceMemoryResource _device_mr
    cdef CudaStreamPool _stream_pool
    cdef Statistics _statistics
    cpdef RmmResourceAdaptor device_mr_adaptor(self)


cdef class OwningDeviceMemoryResource(DeviceMemoryResource):
    cdef any_resource[device_accessible] c_obj

    @staticmethod
    cdef OwningDeviceMemoryResource _create(
        any_resource[device_accessible] resource,
    )
