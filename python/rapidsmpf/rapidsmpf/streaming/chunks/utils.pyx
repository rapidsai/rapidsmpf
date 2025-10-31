# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cpython.object cimport PyObject
from cpython.ref cimport Py_XDECREF


cdef void py_deleter(void *p) noexcept nogil:
    with gil:
        Py_XDECREF(<PyObject*>p)
