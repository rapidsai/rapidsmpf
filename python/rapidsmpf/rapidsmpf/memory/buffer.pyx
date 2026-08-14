# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cpython.buffer cimport PyBUF_WRITE
from cpython.memoryview cimport PyMemoryView_FromMemory
from cython.operator cimport dereference as deref
from libcpp.utility cimport move
from rmm.pylibrmm.stream cimport Stream

from rapidsmpf.memory.buffer_resource cimport BufferResource


cdef class BufferHostView:
    def __cinit__(self, Buffer buf):
        self._buf = buf

    def __enter__(self):
        cdef void* ptr = deref(self._buf._handle).exclusive_data_access()
        try:
            return PyMemoryView_FromMemory(
                <char*>ptr, <Py_ssize_t>deref(self._buf._handle).size, PyBUF_WRITE
            )
        except BaseException:
            deref(self._buf._handle).unlock()
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        deref(self._buf._handle).unlock()
        return False


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
    cdef Buffer from_handle(unique_ptr[cpp_Buffer] handle, BufferResource br, Stream stream):
        cdef Buffer self = Buffer.__new__(Buffer)
        self._handle = move(handle)
        self._br = br
        self._stream = stream
        return self

    @property
    def size(self):
        """Size of the buffer in bytes."""
        return deref(self._handle).size

    @property
    def mem_type(self):
        """Memory type of this buffer."""
        return deref(self._handle).mem_type()

    def host_view(self):
        """Context manager providing exclusive writable host access to the buffer.

        Acquires an exclusive lock on the buffer for the duration of the ``with``
        block, preventing concurrent stream-ordered operations on the C++ side.
        The lock is released (and the returned ``memoryview`` must not be used)
        once the block exits.

        Returns
        -------
        BufferHostView
            A context manager that yields a writable ``memoryview`` of the buffer.

        Raises
        ------
        TypeError
            If the buffer is not a host buffer (``HOST`` or ``PINNED_HOST``).
        std::logic_error
            If the buffer is already locked or a stream-ordered write is still
            in flight (``is_latest_write_done() == False``).

        Examples
        --------
        >>> with buf.host_view() as mv:
        ...     mv[:] = b"\\x00" * buf.size
        """
        if deref(self._handle).mem_type() not in {
            MemoryType.HOST, MemoryType.PINNED_HOST
        }:
            raise TypeError(
                "host_view() is only supported for host buffers "
                "(MemoryType.HOST or MemoryType.PINNED_HOST)"
            )
        return BufferHostView(self)
