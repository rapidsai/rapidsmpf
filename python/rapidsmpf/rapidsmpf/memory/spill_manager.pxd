# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libc.stddef cimport size_t
from libc.stdint cimport int64_t
from libcpp.memory cimport shared_ptr

from rapidsmpf._detail.exception_handling cimport ex_handler


cdef extern from "<rapidsmpf/memory/spill_manager.hpp>" nogil:
    cdef cppclass cpp_SpillFunction "rapidsmpf::SpillManager::SpillFunction":
        pass

    cdef cppclass cpp_SpillManager "rapidsmpf::SpillManager":
        size_t add_spill_function(
            cpp_SpillFunction spill_function, int priority
        ) except +ex_handler
        void remove_spill_function(
            size_t function_id
        ) except +ex_handler
        size_t spill(size_t amount) except +ex_handler
        size_t spill_to_make_headroom(int64_t headroom) except +ex_handler


cdef class SpillManager:
    cdef cpp_SpillManager *_handle
    cdef object _br
    cdef dict _spill_functions
