# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Default values for every configuration option recognised by the
``from_options`` factories and option-driven constructors.

These constants are read directly from the C++ ``rapidsmpf::defaults``
namespace at module import time, so Python and C++ are guaranteed to agree.

Constants are organised by the module that owns the option, mirroring the
sub-namespaces in ``cpp/include/rapidsmpf/defaults.hpp``:

============================  ====================  ==============================
Constant                      Type                  Recognised by option
============================  ====================  ==============================
DEFAULT_STATISTICS            str                   ``statistics``
DEFAULT_PINNED_MEMORY         bool                  ``pinned_memory``
DEFAULT_PINNED_INITIAL_       str                   ``pinned_initial_pool_size``
POOL_SIZE
DEFAULT_PINNED_MAX_POOL_SIZE  str                   ``pinned_max_pool_size``
DEFAULT_SPILL_DEVICE_LIMIT    str                   ``spill_device_limit``
DEFAULT_PERIODIC_SPILL_CHECK  str                   ``periodic_spill_check``
DEFAULT_NUM_STREAMS           int                   ``num_streams``
DEFAULT_NUM_STREAMING_        int                   ``num_streaming_threads``
THREADS
DEFAULT_MEMORY_RESERVE_       str                   ``memory_reserve_timeout``
TIMEOUT
DEFAULT_LOG                   str                   ``log``
DEFAULT_UCXX_PROGRESS_MODE    str                   ``ucxx_progress_mode``
============================  ====================  ==============================
"""

from libc.stdint cimport uint32_t
from libcpp cimport bool as bool_t
from libcpp.string cimport string

cdef extern from "<rapidsmpf/defaults.hpp>" \
        namespace "rapidsmpf::defaults" nogil:
    pass

cdef extern from "<rapidsmpf/defaults.hpp>" \
        namespace "rapidsmpf::defaults::pinned_memory" nogil:
    cdef bool_t _Enabled "rapidsmpf::defaults::pinned_memory::Enabled"

cdef extern from "<rapidsmpf/defaults.hpp>" \
        namespace "rapidsmpf::defaults::buffer_resource" nogil:
    cdef size_t _NumStreams "rapidsmpf::defaults::buffer_resource::NumStreams"

cdef extern from "<rapidsmpf/defaults.hpp>" \
        namespace "rapidsmpf::defaults::streaming" nogil:
    cdef uint32_t _NumStreamingThreads \
        "rapidsmpf::defaults::streaming::NumStreamingThreads"

# Cython has no native string_view binding, so we expose each string_view
# default through a tiny inline helper that copies it into a std::string.
cdef extern from *:
    """
    #include <string>
    #include <rapidsmpf/defaults.hpp>
    namespace {
    inline std::string _rapidsmpf_default_statistics() {
        return std::string{rapidsmpf::defaults::statistics::Enabled};
    }
    inline std::string _rapidsmpf_default_pinned_initial_pool_size() {
        return std::string{
            rapidsmpf::defaults::pinned_memory::InitialPoolSizeFactor
        };
    }
    inline std::string _rapidsmpf_default_pinned_max_pool_size() {
        return std::string{
            rapidsmpf::defaults::pinned_memory::MaxPoolSizeFactor
        };
    }
    inline std::string _rapidsmpf_default_spill_device_limit() {
        return std::string{rapidsmpf::defaults::buffer_resource::SpillDeviceLimit};
    }
    inline std::string _rapidsmpf_default_periodic_spill_check() {
        return std::string{
            rapidsmpf::defaults::buffer_resource::PeriodicSpillCheck
        };
    }
    inline std::string _rapidsmpf_default_memory_reserve_timeout() {
        return std::string{
            rapidsmpf::defaults::streaming::MemoryReserveTimeout
        };
    }
    inline std::string _rapidsmpf_default_log() {
        return std::string{rapidsmpf::defaults::communicator::Log};
    }
    inline std::string _rapidsmpf_default_ucxx_progress_mode() {
        return std::string{rapidsmpf::defaults::ucxx::ProgressMode};
    }
    }  // namespace
    """
    string _cpp_default_statistics "_rapidsmpf_default_statistics"() nogil
    string _cpp_default_pinned_initial_pool_size \
        "_rapidsmpf_default_pinned_initial_pool_size"() nogil
    string _cpp_default_pinned_max_pool_size \
        "_rapidsmpf_default_pinned_max_pool_size"() nogil
    string _cpp_default_spill_device_limit \
        "_rapidsmpf_default_spill_device_limit"() nogil
    string _cpp_default_periodic_spill_check \
        "_rapidsmpf_default_periodic_spill_check"() nogil
    string _cpp_default_memory_reserve_timeout \
        "_rapidsmpf_default_memory_reserve_timeout"() nogil
    string _cpp_default_log "_rapidsmpf_default_log"() nogil
    string _cpp_default_ucxx_progress_mode \
        "_rapidsmpf_default_ucxx_progress_mode"() nogil


cdef _decode(string s):
    return s.decode("utf-8")


DEFAULT_STATISTICS: str = _decode(_cpp_default_statistics())
DEFAULT_PINNED_MEMORY: bool = bool(_Enabled)
DEFAULT_PINNED_INITIAL_POOL_SIZE: str = _decode(
    _cpp_default_pinned_initial_pool_size()
)
DEFAULT_PINNED_MAX_POOL_SIZE: str = _decode(
    _cpp_default_pinned_max_pool_size()
)
DEFAULT_SPILL_DEVICE_LIMIT: str = _decode(_cpp_default_spill_device_limit())
DEFAULT_PERIODIC_SPILL_CHECK: str = _decode(_cpp_default_periodic_spill_check())
DEFAULT_NUM_STREAMS: int = int(_NumStreams)
DEFAULT_NUM_STREAMING_THREADS: int = int(_NumStreamingThreads)
DEFAULT_MEMORY_RESERVE_TIMEOUT: str = _decode(
    _cpp_default_memory_reserve_timeout()
)
DEFAULT_LOG: str = _decode(_cpp_default_log())
DEFAULT_UCXX_PROGRESS_MODE: str = _decode(_cpp_default_ucxx_progress_mode())


__all__ = [
    "DEFAULT_STATISTICS",
    "DEFAULT_PINNED_MEMORY",
    "DEFAULT_PINNED_INITIAL_POOL_SIZE",
    "DEFAULT_PINNED_MAX_POOL_SIZE",
    "DEFAULT_SPILL_DEVICE_LIMIT",
    "DEFAULT_PERIODIC_SPILL_CHECK",
    "DEFAULT_NUM_STREAMS",
    "DEFAULT_NUM_STREAMING_THREADS",
    "DEFAULT_MEMORY_RESERVE_TIMEOUT",
    "DEFAULT_LOG",
    "DEFAULT_UCXX_PROGRESS_MODE",
]
