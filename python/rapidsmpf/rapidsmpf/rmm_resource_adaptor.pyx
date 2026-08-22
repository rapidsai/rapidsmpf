# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference as deref
from libc.stdint cimport uint64_t
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from rapidsmpf.memory.scoped_memory_record cimport ScopedMemoryRecord


cdef class RmmResourceAdaptor(DeviceMemoryResource):
    """
    A RMM memory resource adaptor tailored to RapidsMPF.

    Wraps a primary device memory resource and adds memory usage tracking
    (lifetime stats plus per-thread scoped records).

    .. rubric:: Construction

    This class cannot be constructed directly. A usable ``RmmResourceAdaptor``
    is always owned by a :class:`~rapidsmpf.memory.buffer_resource.BufferResource`
    (which installs the back-reference that makes the adaptor copyable). To obtain
    one, create a ``BufferResource`` from a device memory resource and call
    :meth:`~rapidsmpf.memory.buffer_resource.BufferResource.device_mr_adaptor`:

    >>> br = BufferResource(rmm.mr.CudaMemoryResource())
    >>> mr = br.device_mr_adaptor()

    The returned adaptor holds shared ownership of its owning ``BufferResource``,
    so it (and any copy of it) keeps the ``BufferResource`` alive.
    """
    def __init__(self, *args, **kwargs):
        raise TypeError(
            "RmmResourceAdaptor cannot be constructed directly; obtain one from "
            "BufferResource.device_mr_adaptor()"
        )

    def __dealloc__(self):
        with nogil:
            self.c_obj.reset()

    cdef cpp_RmmResourceAdaptor* get_handle(self):
        return self.c_obj.get()

    @staticmethod
    cdef RmmResourceAdaptor _from_cpp(const cpp_RmmResourceAdaptor& src):
        """
        Create a Python ``RmmResourceAdaptor`` by copying a back-ref'd C++ adaptor.

        The copy acquires shared ownership of the owning ``BufferResource``,
        keeping it alive for the lifetime of the returned Python object.

        Parameters
        ----------
        src
            The C++ ``RmmResourceAdaptor`` to copy from. Must have a back-reference
            installed (i.e. it must have been obtained from a ``BufferResource``);
            otherwise a ``std::bad_weak_ptr`` is raised.

        Returns
        -------
        A new Python ``RmmResourceAdaptor`` wrapping the copied C++ adaptor.
        """
        cdef RmmResourceAdaptor ret = RmmResourceAdaptor.__new__(RmmResourceAdaptor)
        ret.c_obj.reset(new cpp_RmmResourceAdaptor(src))
        ret.c_ref = make_device_async_resource_ref(deref(ret.c_obj))
        return ret

    def get_main_record(self):
        """Returns a copy of the main memory record.

        The main record tracks memory statistics for the lifetime of the resource.

        Returns
        -------
        A copy of the current main memory record.
        """
        cdef cpp_RmmResourceAdaptor* mr = self.get_handle()
        cdef cpp_ScopedMemoryRecord ret
        with nogil:
            ret = deref(mr).get_main_record()
        return ScopedMemoryRecord.from_handle(ret)

    @property
    def current_allocated(self) -> int:
        """Get the total number of currently allocated bytes.

        This includes both allocations on the primary and fallback memory resources.

        Returns
        -------
        Total number of currently allocated bytes.
        """
        cdef cpp_RmmResourceAdaptor* mr = self.get_handle()
        cdef uint64_t ret
        with nogil:
            ret = deref(mr).current_allocated()
        return ret
