# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cpython.object cimport PyObject
from cpython.ref cimport Py_XDECREF


cdef void py_deleter(void *p) noexcept nogil:
    """
    Deletion callback for pyobjects stored in an OwningWrapper.

    Parameters
    ----------
    p
        Pointer to PyObject to delete (may be null)

    Notes
    -----
    Use this as the deleter object when constructing an OwningWrapper.
    """
    with gil:
        Py_XDECREF(<PyObject*>p)
