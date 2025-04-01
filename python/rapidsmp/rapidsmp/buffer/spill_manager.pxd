# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport shared_ptr


cdef extern from "<rapidsmp/buffer/spill_manager.hpp>" nogil:
    cdef cppclass cpp_SpillFunction "rapidsmp::SpillManager::SpillFunction":
        pass

    cdef cppclass cpp_SpillManager "rapidsmp::SpillManager":
        size_t add_spill_function(
            cpp_SpillFunction spill_function, int priority
        ) except +
        void remove_spill_function(
            size_t function_id
        ) except +
        size_t spill(size_t amount) except +


cdef class SpillManager:
    cdef cpp_SpillManager *_handle
    cdef object _br
    cdef dict _spill_functions
