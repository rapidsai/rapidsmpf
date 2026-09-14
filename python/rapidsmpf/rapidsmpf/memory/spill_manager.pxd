# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stddef cimport size_t
from libc.stdint cimport int64_t, uint8_t, uint64_t
from libcpp.memory cimport shared_ptr
from libcpp.optional cimport optional
from libcpp.vector cimport vector

from rapidsmpf._detail.exception_handling cimport ex_handler
from rapidsmpf.memory.buffer cimport MemoryType


cdef extern from "<rapidsmpf/memory/spill_manager.hpp>" nogil:
    cpdef enum class SpillReason "rapidsmpf::SpillReason" (uint8_t):
        PERIODIC
        RESERVATION
        EAGER
        EXPLICIT

    cdef cppclass cpp_SpillAttemptRecord "rapidsmpf::SpillAttemptRecord":
        uint64_t event_id
        uint64_t attempt_id
        int64_t start_ns
        size_t requested_bytes
        SpillReason reason
        bint is_background
        optional[uint64_t] evictor

    cdef cppclass cpp_SpillTransferRecord "rapidsmpf::SpillTransferRecord":
        uint64_t event_id
        uint64_t attempt_id
        int64_t submission_start_ns
        int64_t submission_end_ns
        size_t submitted_bytes
        MemoryType source
        MemoryType destination
        SpillReason reason
        bint is_background
        optional[uint64_t] evictor
        optional[uint64_t] buffer_owner
        bint is_success

    cdef cppclass cpp_SpillEventCollector "rapidsmpf::SpillEventCollector":
        void enable(size_t capacity) except +ex_handler
        void disable() noexcept
        bint enabled() noexcept
        uint64_t sequence() noexcept
        uint64_t dropped_events() noexcept
        vector[cpp_SpillAttemptRecord] read_attempts(
            uint64_t begin_sequence, uint64_t end_sequence
        ) except +ex_handler
        vector[cpp_SpillTransferRecord] read_transfers(
            uint64_t begin_sequence, uint64_t end_sequence
        ) except +ex_handler

    cdef cppclass cpp_SpillFunction "rapidsmpf::SpillManager::SpillFunction":
        pass

    cdef cppclass cpp_SpillManager "rapidsmpf::SpillManager":
        size_t add_spill_function(
            cpp_SpillFunction spill_function,
            int priority,
            optional[uint64_t] buffer_owner
        ) except +ex_handler
        void remove_spill_function(
            size_t function_id
        ) except +ex_handler
        size_t spill(
            size_t amount, SpillReason reason, optional[uint64_t] evictor
        ) except +ex_handler
        size_t spill_to_make_headroom(int64_t headroom) except +ex_handler
        cpp_SpillEventCollector& event_collector() noexcept


cdef class SpillManager:
    cdef cpp_SpillManager *_handle
    cdef object _br
    cdef dict _spill_functions
