# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Compile-time keys and default values for every configuration option recognised
by the ``from_options`` factories and option-driven constructors.

Each option exposes a module-level constant holding the lookup string passed
to :class:`rapidsmpf.config.Options`. Default values are collected in the
read-only :data:`DEFAULTS` mapping, keyed by the same key string.

Names are prefixed with the originating C++ namespace (``STATISTICS_``,
``PINNED_MEMORY_``, ``BUFFER_RESOURCE_``, ``STREAMING_``, ``COMMUNICATOR_``,
``UCXX_``).

Values are sourced from the C++ ``<rapidsmpf/config.hpp>`` header at module
import time, so Python and C++ cannot drift out of sync.

Example:
    >>> from rapidsmpf import config_defaults
    >>> config_defaults.BUFFER_RESOURCE_NUM_STREAMS
    'num_streams'
    >>> config_defaults.DEFAULTS[config_defaults.BUFFER_RESOURCE_NUM_STREAMS]
    '16'
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final


# Pull every key and default out of the C++ header. Member-access expressions
# (e.g. ``EnabledOption.key``) cannot appear directly in a Cython
# ``cdef extern`` alias, so a tiny helper namespace re-exposes each as a
# plain identifier with a stable spelling. Both ``key`` and ``default_val``
# are owning ``std::string``s with static storage duration; ``.c_str()``
# yields a long-lived, null-terminated ``const char*`` suitable for Cython
# byte decoding.
cdef extern from *:
    """
    #include <rapidsmpf/config.hpp>
    namespace rapidsmpf_options_py {
    // Exposes `k_<SUFFIX>` and `d_<SUFFIX>` as `const char*` aliases for the
    // key and default-value strings of `rapidsmpf::NS::OPT`.
    #define RMPF_OPT(SUFFIX, NS, OPT) \\
        inline char const* const k_##SUFFIX = rapidsmpf::NS::OPT.key.c_str(); \\
        inline char const* const d_##SUFFIX = rapidsmpf::NS::OPT.default_val.c_str();

    RMPF_OPT(statistics, statistics, EnabledOption)
    RMPF_OPT(pinned_memory, pinned_memory, EnabledOption)
    RMPF_OPT(pinned_initial_pool_size, pinned_memory, InitialPoolSizeOption)
    RMPF_OPT(pinned_max_pool_size, pinned_memory, MaxPoolSizeOption)
    RMPF_OPT(spill_device_limit, buffer_resource, SpillDeviceLimitOption)
    RMPF_OPT(periodic_spill_check, buffer_resource, PeriodicSpillCheckOption)
    RMPF_OPT(num_streams, buffer_resource, NumStreamsOption)
    RMPF_OPT(num_streaming_threads, streaming, NumStreamingThreadsOption)
    RMPF_OPT(memory_reserve_timeout, streaming, MemoryReserveTimeoutOption)
    RMPF_OPT(allow_overbooking_by_default,
             streaming, AllowOverbookingByDefaultOption)
    RMPF_OPT(log, communicator, LogOption)
    RMPF_OPT(ucxx_progress_mode, ucxx, ProgressModeOption)

    #undef RMPF_OPT
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
    const char* _k_allow_overbooking_by_default \
        "rapidsmpf_options_py::k_allow_overbooking_by_default"
    const char* _k_log "rapidsmpf_options_py::k_log"
    const char* _k_ucxx_progress_mode \
        "rapidsmpf_options_py::k_ucxx_progress_mode"

    const char* _d_statistics "rapidsmpf_options_py::d_statistics"
    const char* _d_pinned_memory "rapidsmpf_options_py::d_pinned_memory"
    const char* _d_pinned_initial_pool_size \
        "rapidsmpf_options_py::d_pinned_initial_pool_size"
    const char* _d_pinned_max_pool_size \
        "rapidsmpf_options_py::d_pinned_max_pool_size"
    const char* _d_spill_device_limit \
        "rapidsmpf_options_py::d_spill_device_limit"
    const char* _d_periodic_spill_check \
        "rapidsmpf_options_py::d_periodic_spill_check"
    const char* _d_num_streams "rapidsmpf_options_py::d_num_streams"
    const char* _d_num_streaming_threads \
        "rapidsmpf_options_py::d_num_streaming_threads"
    const char* _d_memory_reserve_timeout \
        "rapidsmpf_options_py::d_memory_reserve_timeout"
    const char* _d_allow_overbooking_by_default \
        "rapidsmpf_options_py::d_allow_overbooking_by_default"
    const char* _d_log "rapidsmpf_options_py::d_log"
    const char* _d_ucxx_progress_mode \
        "rapidsmpf_options_py::d_ucxx_progress_mode"


cdef _decode(const char* s):
    return (<bytes>s).decode("utf-8")


# Options for `rapidsmpf::statistics`.
STATISTICS_ENABLED: Final[str] = _decode(_k_statistics)

# Options for `rapidsmpf::pinned_memory`.
PINNED_MEMORY_ENABLED: Final[str] = _decode(_k_pinned_memory)
PINNED_MEMORY_INITIAL_POOL_SIZE: Final[str] = _decode(
    _k_pinned_initial_pool_size
)
PINNED_MEMORY_MAX_POOL_SIZE: Final[str] = _decode(_k_pinned_max_pool_size)

# Options for `rapidsmpf::buffer_resource`.
BUFFER_RESOURCE_SPILL_DEVICE_LIMIT: Final[str] = _decode(_k_spill_device_limit)
BUFFER_RESOURCE_PERIODIC_SPILL_CHECK: Final[str] = _decode(_k_periodic_spill_check)
BUFFER_RESOURCE_NUM_STREAMS: Final[str] = _decode(_k_num_streams)

# Options for `rapidsmpf::streaming`.
STREAMING_NUM_STREAMING_THREADS: Final[str] = _decode(_k_num_streaming_threads)
STREAMING_MEMORY_RESERVE_TIMEOUT: Final[str] = _decode(_k_memory_reserve_timeout)
STREAMING_ALLOW_OVERBOOKING_BY_DEFAULT: Final[str] = _decode(
    _k_allow_overbooking_by_default
)

# Options for `rapidsmpf::communicator` (consumed by `Communicator::Logger`).
COMMUNICATOR_LOG: Final[str] = _decode(_k_log)

# Options for `rapidsmpf::ucxx` (consumed by `rapidsmpf::ucxx::init`).
UCXX_PROGRESS_MODE: Final[str] = _decode(_k_ucxx_progress_mode)


# Read-only mapping from option key to default value. Values are the raw
# string representation sourced from C++ (the canonical form, since options
# are parsed from strings at runtime); callers convert to typed values as
# needed using the same parser they apply to user-supplied option strings.
DEFAULTS: Final[Mapping[str, str]] = MappingProxyType({
    STATISTICS_ENABLED: _decode(_d_statistics),
    PINNED_MEMORY_ENABLED: _decode(_d_pinned_memory),
    PINNED_MEMORY_INITIAL_POOL_SIZE: _decode(_d_pinned_initial_pool_size),
    PINNED_MEMORY_MAX_POOL_SIZE: _decode(_d_pinned_max_pool_size),
    BUFFER_RESOURCE_SPILL_DEVICE_LIMIT: _decode(_d_spill_device_limit),
    BUFFER_RESOURCE_PERIODIC_SPILL_CHECK: _decode(_d_periodic_spill_check),
    BUFFER_RESOURCE_NUM_STREAMS: _decode(_d_num_streams),
    STREAMING_NUM_STREAMING_THREADS: _decode(_d_num_streaming_threads),
    STREAMING_MEMORY_RESERVE_TIMEOUT: _decode(_d_memory_reserve_timeout),
    STREAMING_ALLOW_OVERBOOKING_BY_DEFAULT: _decode(
        _d_allow_overbooking_by_default
    ),
    COMMUNICATOR_LOG: _decode(_d_log),
    UCXX_PROGRESS_MODE: _decode(_d_ucxx_progress_mode),
})


__all__ = [
    "STATISTICS_ENABLED",
    "PINNED_MEMORY_ENABLED",
    "PINNED_MEMORY_INITIAL_POOL_SIZE",
    "PINNED_MEMORY_MAX_POOL_SIZE",
    "BUFFER_RESOURCE_SPILL_DEVICE_LIMIT",
    "BUFFER_RESOURCE_PERIODIC_SPILL_CHECK",
    "BUFFER_RESOURCE_NUM_STREAMS",
    "STREAMING_NUM_STREAMING_THREADS",
    "STREAMING_MEMORY_RESERVE_TIMEOUT",
    "STREAMING_ALLOW_OVERBOOKING_BY_DEFAULT",
    "COMMUNICATOR_LOG",
    "UCXX_PROGRESS_MODE",
    "DEFAULTS",
]
