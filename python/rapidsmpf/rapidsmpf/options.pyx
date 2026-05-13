# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Compile-time descriptors for every configuration option recognised by the
``from_options`` factories and option-driven constructors.

Each descriptor is an :class:`OptionDescriptor` exposing ``key`` (the lookup
string passed to ``Options``) and ``default_value`` (the value used when the
option is unset). Descriptors are grouped under module sub-namespaces that
mirror the C++ layout (``rapidsmpf::statistics``, ``rapidsmpf::pinned_memory``,
``rapidsmpf::buffer_resource``, ``rapidsmpf::streaming``,
``rapidsmpf::communicator``, ``rapidsmpf::ucxx``).

Values are sourced from the C++ ``<rapidsmpf/options.hpp>`` header at module
import time, so Python and C++ cannot drift out of sync.

Example:
    >>> from rapidsmpf import options
    >>> options.buffer_resource.NumStreamsOption.key
    'num_streams'
    >>> options.buffer_resource.NumStreamsOption.default_value
    16
"""

from dataclasses import dataclass
from typing import Generic, TypeVar

from libc.stdint cimport uint32_t
from libcpp cimport bool as bool_t


T = TypeVar("T")


@dataclass(frozen=True)
class OptionDescriptor(Generic[T]):
    """Lookup key paired with the default value for a single option."""

    key: str
    default_value: T


# Pull every key and default out of the C++ header. Member-access expressions
# (e.g. ``EnabledOption.key``) cannot appear directly in a Cython
# ``cdef extern`` alias, so a tiny helper namespace re-exposes each as a
# plain identifier with a stable spelling.
cdef extern from *:
    """
    #include <rapidsmpf/options.hpp>
    namespace rapidsmpf_options_py {
    // keys
    inline constexpr const char* k_statistics =
        rapidsmpf::statistics::EnabledOption.key;
    inline constexpr const char* k_pinned_memory =
        rapidsmpf::pinned_memory::EnabledOption.key;
    inline constexpr const char* k_pinned_initial_pool_size =
        rapidsmpf::pinned_memory::InitialPoolSizeFactorOption.key;
    inline constexpr const char* k_pinned_max_pool_size =
        rapidsmpf::pinned_memory::MaxPoolSizeFactorOption.key;
    inline constexpr const char* k_spill_device_limit =
        rapidsmpf::buffer_resource::SpillDeviceLimitOption.key;
    inline constexpr const char* k_periodic_spill_check =
        rapidsmpf::buffer_resource::PeriodicSpillCheckOption.key;
    inline constexpr const char* k_num_streams =
        rapidsmpf::buffer_resource::NumStreamsOption.key;
    inline constexpr const char* k_num_streaming_threads =
        rapidsmpf::streaming::NumStreamingThreadsOption.key;
    inline constexpr const char* k_memory_reserve_timeout =
        rapidsmpf::streaming::MemoryReserveTimeoutOption.key;
    inline constexpr const char* k_log =
        rapidsmpf::communicator::LogOption.key;
    inline constexpr const char* k_ucxx_progress_mode =
        rapidsmpf::ucxx::ProgressModeOption.key;

    // string-view defaults exposed as null-terminated `const char*`
    inline constexpr const char* d_statistics =
        rapidsmpf::statistics::EnabledOption.default_value.data();
    inline constexpr const char* d_pinned_initial_pool_size =
        rapidsmpf::pinned_memory::InitialPoolSizeFactorOption.default_value.data();
    inline constexpr const char* d_pinned_max_pool_size =
        rapidsmpf::pinned_memory::MaxPoolSizeFactorOption.default_value.data();
    inline constexpr const char* d_spill_device_limit =
        rapidsmpf::buffer_resource::SpillDeviceLimitOption.default_value.data();
    inline constexpr const char* d_periodic_spill_check =
        rapidsmpf::buffer_resource::PeriodicSpillCheckOption.default_value.data();
    inline constexpr const char* d_memory_reserve_timeout =
        rapidsmpf::streaming::MemoryReserveTimeoutOption.default_value.data();
    inline constexpr const char* d_log =
        rapidsmpf::communicator::LogOption.default_value.data();
    inline constexpr const char* d_ucxx_progress_mode =
        rapidsmpf::ucxx::ProgressModeOption.default_value.data();

    // typed defaults
    inline constexpr bool d_pinned_memory =
        rapidsmpf::pinned_memory::EnabledOption.default_value;
    inline constexpr size_t d_num_streams =
        rapidsmpf::buffer_resource::NumStreamsOption.default_value;
    inline constexpr uint32_t d_num_streaming_threads =
        rapidsmpf::streaming::NumStreamingThreadsOption.default_value;
    }  // namespace rapidsmpf_options_py
    """
    const char* _k_statistics "rapidsmpf_options_py::k_statistics"
    const char* _k_pinned_memory "rapidsmpf_options_py::k_pinned_memory"
    const char* _k_pinned_initial_pool_size \
        "rapidsmpf_options_py::k_pinned_initial_pool_size"
    const char* _k_pinned_max_pool_size \
        "rapidsmpf_options_py::k_pinned_max_pool_size"
    const char* _k_spill_device_limit \
        "rapidsmpf_options_py::k_spill_device_limit"
    const char* _k_periodic_spill_check \
        "rapidsmpf_options_py::k_periodic_spill_check"
    const char* _k_num_streams "rapidsmpf_options_py::k_num_streams"
    const char* _k_num_streaming_threads \
        "rapidsmpf_options_py::k_num_streaming_threads"
    const char* _k_memory_reserve_timeout \
        "rapidsmpf_options_py::k_memory_reserve_timeout"
    const char* _k_log "rapidsmpf_options_py::k_log"
    const char* _k_ucxx_progress_mode \
        "rapidsmpf_options_py::k_ucxx_progress_mode"

    const char* _d_statistics "rapidsmpf_options_py::d_statistics"
    const char* _d_pinned_initial_pool_size \
        "rapidsmpf_options_py::d_pinned_initial_pool_size"
    const char* _d_pinned_max_pool_size \
        "rapidsmpf_options_py::d_pinned_max_pool_size"
    const char* _d_spill_device_limit \
        "rapidsmpf_options_py::d_spill_device_limit"
    const char* _d_periodic_spill_check \
        "rapidsmpf_options_py::d_periodic_spill_check"
    const char* _d_memory_reserve_timeout \
        "rapidsmpf_options_py::d_memory_reserve_timeout"
    const char* _d_log "rapidsmpf_options_py::d_log"
    const char* _d_ucxx_progress_mode \
        "rapidsmpf_options_py::d_ucxx_progress_mode"

    bool_t _d_pinned_memory "rapidsmpf_options_py::d_pinned_memory"
    size_t _d_num_streams "rapidsmpf_options_py::d_num_streams"
    uint32_t _d_num_streaming_threads \
        "rapidsmpf_options_py::d_num_streaming_threads"


cdef _decode(const char* s):
    return (<bytes>s).decode("utf-8")


class statistics:
    """Options for :py:meth:`rapidsmpf.statistics.Statistics.from_options`."""

    EnabledOption: OptionDescriptor[str] = OptionDescriptor(
        key=_decode(_k_statistics),
        default_value=_decode(_d_statistics),
    )


class pinned_memory:
    """Options for
    :py:meth:`rapidsmpf.memory.pinned_memory_resource.PinnedMemoryResource.from_options`.
    """

    EnabledOption: OptionDescriptor[bool] = OptionDescriptor(
        key=_decode(_k_pinned_memory),
        default_value=bool(_d_pinned_memory),
    )
    InitialPoolSizeFactorOption: OptionDescriptor[str] = OptionDescriptor(
        key=_decode(_k_pinned_initial_pool_size),
        default_value=_decode(_d_pinned_initial_pool_size),
    )
    MaxPoolSizeFactorOption: OptionDescriptor[str] = OptionDescriptor(
        key=_decode(_k_pinned_max_pool_size),
        default_value=_decode(_d_pinned_max_pool_size),
    )


class buffer_resource:
    """Options for the buffer resource and its helpers."""

    SpillDeviceLimitOption: OptionDescriptor[str] = OptionDescriptor(
        key=_decode(_k_spill_device_limit),
        default_value=_decode(_d_spill_device_limit),
    )
    PeriodicSpillCheckOption: OptionDescriptor[str] = OptionDescriptor(
        key=_decode(_k_periodic_spill_check),
        default_value=_decode(_d_periodic_spill_check),
    )
    NumStreamsOption: OptionDescriptor[int] = OptionDescriptor(
        key=_decode(_k_num_streams),
        default_value=int(_d_num_streams),
    )


class streaming:
    """Options for the streaming subsystem."""

    NumStreamingThreadsOption: OptionDescriptor[int] = OptionDescriptor(
        key=_decode(_k_num_streaming_threads),
        default_value=int(_d_num_streaming_threads),
    )
    MemoryReserveTimeoutOption: OptionDescriptor[str] = OptionDescriptor(
        key=_decode(_k_memory_reserve_timeout),
        default_value=_decode(_d_memory_reserve_timeout),
    )


class communicator:
    """Options consumed by ``rapidsmpf::Communicator::Logger``."""

    LogOption: OptionDescriptor[str] = OptionDescriptor(
        key=_decode(_k_log),
        default_value=_decode(_d_log),
    )


class ucxx:
    """Options consumed by ``rapidsmpf::ucxx::init``."""

    ProgressModeOption: OptionDescriptor[str] = OptionDescriptor(
        key=_decode(_k_ucxx_progress_mode),
        default_value=_decode(_d_ucxx_progress_mode),
    )


__all__ = [
    "OptionDescriptor",
    "statistics",
    "pinned_memory",
    "buffer_resource",
    "streaming",
    "communicator",
    "ucxx",
]
