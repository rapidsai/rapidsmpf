# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

cdef extern from "<rapidsmpf/owning_wrapper.hpp>" nogil:
    cdef cppclass cpp_OwningWrapper "rapidsmpf::OwningWrapper":
        cpp_OwningWrapper(void *, void(*)(void*)) noexcept
        void* release() noexcept
