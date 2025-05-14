# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0


cdef class Options:
    def __cinit__(self):
        with nogil:
            self._handle = cpp_Options()
