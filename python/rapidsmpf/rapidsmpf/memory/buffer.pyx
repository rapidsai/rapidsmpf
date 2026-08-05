# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference as deref
from libc.stddef cimport size_t
from libcpp.utility cimport move


cdef class Buffer:
    """A stream-ordered host or device memory buffer managed by a
    :class:`~rapidsmpf.memory.buffer_resource.BufferResource`.

    Buffers are not constructed directly; use
    :meth:`~rapidsmpf.memory.buffer_resource.BufferResource.make_buffer` to obtain one.
    """

    def __init__(self):
        raise ValueError("Buffer must be created via BufferResource.make_buffer")

    def __dealloc__(self):
        with nogil:
            self._handle.reset()

    @staticmethod
    cdef Buffer from_handle(unique_ptr[cpp_Buffer] handle):
        cdef Buffer self = Buffer.__new__(Buffer)
        self._handle = move(handle)
        return self

    @property
    def size(self):
        """Size of the buffer in bytes."""
        return deref(self._handle).size

    @property
    def mem_type(self):
        """Memory type of this buffer."""
        return deref(self._handle).mem_type()

    def __getbuffer__(self, Py_buffer* view, int flags):
        if deref(self._handle).mem_type() != MemoryType.PINNED_HOST:
            raise TypeError(
                "buffer protocol is only supported for PINNED_HOST buffers"
            )
        cdef size_t nbytes = deref(self._handle).size
        cdef const char* ptr = deref(self._handle).data()
        view.buf = <void*>ptr
        view.len = nbytes
        view.readonly = 0
        view.format = 'B'
        view.ndim = 1
        view.shape = &view.len
        view.strides = NULL
        view.suboffsets = NULL
        view.itemsize = 1
        view.internal = NULL
        view.obj = self

    def __releasebuffer__(self, Py_buffer* view):
        pass
