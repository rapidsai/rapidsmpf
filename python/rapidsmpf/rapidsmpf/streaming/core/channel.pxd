# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stddef cimport size_t
from libc.stdint cimport uint32_t, uint64_t
from libcpp.memory cimport shared_ptr

from rapidsmpf._detail.exception_handling cimport ex_handler
from rapidsmpf.memory.buffer cimport MemoryType as cpp_MemoryType


cdef extern from "<rapidsmpf/streaming/core/channel.hpp>" nogil:
    cdef cppclass cpp_Channel"rapidsmpf::streaming::Channel":
        cpp_Channel() except +ex_handler
        cpp_ChannelMetricsSnapshot metrics() noexcept

    cdef cppclass cpp_ChannelMetricsSnapshot"rapidsmpf::streaming::Channel::MetricsSnapshot":
        uint32_t message_count
        uint32_t spillable_count

cdef extern from * nogil:
    """
    namespace {
    [[maybe_unused]] std::uint64_t cpp_metrics_send_bytes(
        rapidsmpf::streaming::Channel::MetricsSnapshot const& metrics,
        rapidsmpf::MemoryType mem_type) noexcept {
      return metrics.send_bytes[static_cast<std::size_t>(mem_type)];
    }
    [[maybe_unused]] std::uint64_t cpp_metrics_recv_bytes(
        rapidsmpf::streaming::Channel::MetricsSnapshot const& metrics,
        rapidsmpf::MemoryType mem_type) noexcept {
      return metrics.recv_bytes[static_cast<std::size_t>(mem_type)];
    }
    }
    """
    uint64_t cpp_metrics_send_bytes(const cpp_ChannelMetricsSnapshot &, cpp_MemoryType) noexcept
    uint64_t cpp_metrics_recv_bytes(const cpp_ChannelMetricsSnapshot &, cpp_MemoryType) noexcept


cdef class Channel:
    cdef shared_ptr[cpp_Channel] _handle

    @staticmethod
    cdef from_handle(shared_ptr[cpp_Channel] ch)

cdef class ChannelMetrics:
    cdef cpp_ChannelMetricsSnapshot _handle

    @staticmethod
    cdef from_handle(cpp_ChannelMetricsSnapshot metrics)
