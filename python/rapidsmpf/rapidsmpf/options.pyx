# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Compile-time descriptors for every configuration option recognised by the
``from_options`` factories and option-driven constructors.

Each descriptor is an :class:`OptionDescriptor` exposing ``key`` (the lookup
string passed to ``Options``) and ``default_val`` (the value used when the
option is unset). Descriptors are exposed as module-level constants whose
names are prefixed with the originating C++ namespace
(``rapidsmpf::statistics``, ``rapidsmpf::pinned_memory``,
``rapidsmpf::buffer_resource``, ``rapidsmpf::streaming``,
``rapidsmpf::communicator``, ``rapidsmpf::ucxx``).

Values are sourced from the C++ ``<rapidsmpf/options.hpp>`` header at module
import time, so Python and C++ cannot drift out of sync.

Example:
    >>> from rapidsmpf import options
    >>> options.BufferResourceNumStreamsOption.key
    'num_streams'
    >>> options.BufferResourceNumStreamsOption.default_val
    16
"""

from typing import Generic, NamedTuple, TypeVar

from libc.stdint cimport uint32_t
from libcpp cimport bool as bool_t

T = TypeVar("T")


class OptionDescriptor(NamedTuple, Generic[T]):
    """Lookup key paired with the default value for a single option."""

    key: str
    default_val: T


# Pull every key and default out of the C++ header. Member-access expressions
# (e.g. ``EnabledOption.key``) cannot appear directly in a Cython
# ``cdef extern`` alias, so a tiny helper namespace re-exposes each as a
# plain identifier with a stable spelling. Macros keep the per-option
# boilerplate to a single line.
cdef extern from *:
    """
    #include <rapidsmpf/options.hpp>
    namespace rapidsmpf_options_py {
    // For string-view defaults: exposes `k_<SUFFIX>` and `d_<SUFFIX>`
    // as `const char*` aliases for `rapidsmpf::NS::OPT`.
    #define RMPF_STR_OPT(SUFFIX, NS, OPT) \\
        inline constexpr const char* k_##SUFFIX = rapidsmpf::NS::OPT.key; \\
        inline constexpr const char* d_##SUFFIX = \\
            rapidsmpf::NS::OPT.default_val.data();
    // For non-string defaults: same as RMPF_STR_OPT but `d_<SUFFIX>` has type T.
    #define RMPF_TYPED_OPT(T, SUFFIX, NS, OPT) \\
        inline constexpr const char* k_##SUFFIX = rapidsmpf::NS::OPT.key; \\
        inline constexpr T d_##SUFFIX = rapidsmpf::NS::OPT.default_val;

    RMPF_STR_OPT(statistics, statistics, EnabledOption)
    RMPF_TYPED_OPT(bool, pinned_memory, pinned_memory, EnabledOption)
    RMPF_STR_OPT(pinned_initial_pool_size,
                 pinned_memory, InitialPoolSizeFactorOption)
    RMPF_STR_OPT(pinned_max_pool_size,
                 pinned_memory, MaxPoolSizeFactorOption)
    RMPF_STR_OPT(spill_device_limit,
                 buffer_resource, SpillDeviceLimitOption)
    RMPF_STR_OPT(periodic_spill_check,
                 buffer_resource, PeriodicSpillCheckOption)
    RMPF_TYPED_OPT(std::size_t, num_streams,
                   buffer_resource, NumStreamsOption)
    RMPF_TYPED_OPT(std::uint32_t, num_streaming_threads,
                   streaming, NumStreamingThreadsOption)
    RMPF_STR_OPT(memory_reserve_timeout,
                 streaming, MemoryReserveTimeoutOption)
    RMPF_STR_OPT(log, communicator, LogOption)
    RMPF_STR_OPT(ucxx_progress_mode, ucxx, ProgressModeOption)

    #undef RMPF_STR_OPT
    #undef RMPF_TYPED_OPT
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


# Options for `rapidsmpf::statistics`.
StatisticsEnabledOption: OptionDescriptor[str] = OptionDescriptor(
    key=_decode(_k_statistics),
    default_val=_decode(_d_statistics),
)

# Options for `rapidsmpf::pinned_memory`.
PinnedMemoryEnabledOption: OptionDescriptor[bool] = OptionDescriptor(
    key=_decode(_k_pinned_memory),
    default_val=bool(_d_pinned_memory),
)
PinnedMemoryInitialPoolSizeFactorOption: OptionDescriptor[str] = OptionDescriptor(
    key=_decode(_k_pinned_initial_pool_size),
    default_val=_decode(_d_pinned_initial_pool_size),
)
PinnedMemoryMaxPoolSizeFactorOption: OptionDescriptor[str] = OptionDescriptor(
    key=_decode(_k_pinned_max_pool_size),
    default_val=_decode(_d_pinned_max_pool_size),
)

# Options for `rapidsmpf::buffer_resource`.
BufferResourceSpillDeviceLimitOption: OptionDescriptor[str] = OptionDescriptor(
    key=_decode(_k_spill_device_limit),
    default_val=_decode(_d_spill_device_limit),
)
BufferResourcePeriodicSpillCheckOption: OptionDescriptor[str] = OptionDescriptor(
    key=_decode(_k_periodic_spill_check),
    default_val=_decode(_d_periodic_spill_check),
)
BufferResourceNumStreamsOption: OptionDescriptor[int] = OptionDescriptor(
    key=_decode(_k_num_streams),
    default_val=int(_d_num_streams),
)

# Options for `rapidsmpf::streaming`.
StreamingNumStreamingThreadsOption: OptionDescriptor[int] = OptionDescriptor(
    key=_decode(_k_num_streaming_threads),
    default_val=int(_d_num_streaming_threads),
)
StreamingMemoryReserveTimeoutOption: OptionDescriptor[str] = OptionDescriptor(
    key=_decode(_k_memory_reserve_timeout),
    default_val=_decode(_d_memory_reserve_timeout),
)

# Options for `rapidsmpf::communicator` (consumed by `Communicator::Logger`).
CommunicatorLogOption: OptionDescriptor[str] = OptionDescriptor(
    key=_decode(_k_log),
    default_val=_decode(_d_log),
)

# Options for `rapidsmpf::ucxx` (consumed by `rapidsmpf::ucxx::init`).
UcxxProgressModeOption: OptionDescriptor[str] = OptionDescriptor(
    key=_decode(_k_ucxx_progress_mode),
    default_val=_decode(_d_ucxx_progress_mode),
)


__all__ = [
    "OptionDescriptor",
    "StatisticsEnabledOption",
    "PinnedMemoryEnabledOption",
    "PinnedMemoryInitialPoolSizeFactorOption",
    "PinnedMemoryMaxPoolSizeFactorOption",
    "BufferResourceSpillDeviceLimitOption",
    "BufferResourcePeriodicSpillCheckOption",
    "BufferResourceNumStreamsOption",
    "StreamingNumStreamingThreadsOption",
    "StreamingMemoryReserveTimeoutOption",
    "CommunicatorLogOption",
    "UcxxProgressModeOption",
]
