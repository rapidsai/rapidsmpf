# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cython declarations for the ProgressThread interface."""

from cython.operator cimport dereference as deref
from libc.stddef cimport size_t
from libc.stdint cimport int32_t, int64_t, uint8_t, uint64_t, uintptr_t
from libcpp.functional cimport function
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector

from rapidsmpf._detail.exception_handling cimport ex_handler
from rapidsmpf.memory.buffer cimport MemoryType
from rapidsmpf.statistics cimport cpp_Statistics


cdef extern from "<rapidsmpf/progress_thread.hpp>" nogil:
    cpdef enum class CollectiveKind "rapidsmpf::CollectiveKind"(uint8_t):
        ALLGATHER
        SPARSE_ALLTOALL
        SHUFFLER
        ALLREDUCE

    cdef cppclass cpp_TransferEvent "rapidsmpf::TransferEvent":
        int32_t op_id
        CollectiveKind collective_kind
        int32_t source_rank
        int32_t destination_rank
        uint64_t message_id
        uint64_t metadata_bytes
        uint64_t payload_bytes
        MemoryType destination_memory_type
        int64_t completion_timestamp_ns

    cdef cppclass cpp_ProgressThread "rapidsmpf::ProgressThread":
        cpp_ProgressThread(
            shared_ptr[cpp_Statistics] statistics,
        ) except +ex_handler

        shared_ptr[cpp_Statistics] statistics()
        void enable_transfer_events(size_t capacity) except +ex_handler
        void disable_transfer_events() except +ex_handler
        vector[cpp_TransferEvent] drain_transfer_events() except +ex_handler
        uint64_t dropped_transfer_events() noexcept


cdef class ProgressThread:
    cdef shared_ptr[cpp_ProgressThread] _handle
