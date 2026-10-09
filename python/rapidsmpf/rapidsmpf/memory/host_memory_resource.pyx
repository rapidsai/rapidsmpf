# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int64_t
from libcpp.optional cimport optional
from rmm.pylibrmm.stream cimport Stream

from rapidsmpf._detail.exception_handling cimport ex_handler
from rapidsmpf.memory.scoped_memory_record cimport (ScopedMemoryRecord,
                                                    cpp_ScopedMemoryRecord)


cdef extern from *:
    """
    #include <optional>

    namespace {
    // Copy a back-referenced `HostMemoryResource`. The copy promotes the
    // resource's back-reference, so the result keeps the owning `BufferResource`
    // alive. Throws `std::bad_weak_ptr` if the resource carries no back-reference.
    std::optional<rapidsmpf::HostMemoryResource>
    cpp_copy_host_mr(rapidsmpf::HostMemoryResource const& src) {
        return src;
    }
    }  // namespace
    """
    optional[cpp_HostMemoryResource] cpp_copy_host_mr(
        const cpp_HostMemoryResource&
    ) except +ex_handler nogil


cdef class HostMemoryResource:
    """
    Opaque handle to a tracked pageable-host memory resource.

    Allocations made through this resource are tracked, so the amount of
    currently allocated host memory and its lifetime statistics can be queried.

    .. rubric:: Construction

    This class cannot be constructed directly. The host memory resource is owned
    by a :class:`~rapidsmpf.memory.buffer_resource.BufferResource`. Obtain the
    handle via
    :attr:`~rapidsmpf.memory.buffer_resource.BufferResource.host_mr`.

    The returned handle holds shared ownership of its owning ``BufferResource``,
    so it (and any copy of it) keeps the ``BufferResource`` alive.
    """
    def __init__(self, *args, **kwargs):
        raise TypeError(
            "HostMemoryResource cannot be constructed directly; obtain it via "
            "BufferResource.host_mr"
        )

    def __dealloc__(self):
        with nogil:
            self._handle.reset()

    def allocate(self, size_t nbytes, Stream stream not None) -> int:
        """
        Allocate pageable host memory associated with a CUDA stream.

        Parameters
        ----------
        nbytes
            Number of bytes to allocate.
        stream
            CUDA stream to associate with the allocation.

        Returns
        -------
        Integer address of the allocated memory.
        """
        cdef void* ptr
        with nogil:
            ptr = self._handle.value().allocate(stream.view(), nbytes)
        return <size_t>ptr

    def deallocate(self, size_t ptr, size_t nbytes, Stream stream not None) -> None:
        """
        Deallocate pageable host memory associated with a CUDA stream.

        Parameters
        ----------
        ptr
            Integer address previously returned by :meth:`allocate`.
        nbytes
            Number of bytes originally allocated.
        stream
            CUDA stream associated with the allocation.
        """
        with nogil:
            self._handle.value().deallocate(stream.view(), <void*>ptr, nbytes)

    @property
    def current_allocated(self) -> int:
        """
        Total number of bytes currently allocated through this resource.
        """
        cdef int64_t ret
        with nogil:
            ret = self._handle.value().current_allocated()
        return ret

    def get_main_memory_record(self):
        """
        Returns a copy of the main memory record.

        The main record tracks memory statistics for the lifetime of the resource.

        Returns
        -------
        A copy of the current main memory record.
        """
        cdef cpp_ScopedMemoryRecord ret
        with nogil:
            ret = self._handle.value().get_main_memory_record()
        return ScopedMemoryRecord.from_handle(ret)

    @staticmethod
    cdef HostMemoryResource from_handle(
        const cpp_HostMemoryResource& handle
    ):
        """
        Create a Python ``HostMemoryResource`` by copying a back-ref'd C++ handle.

        The copy acquires shared ownership of the owning ``BufferResource``,
        keeping it alive for the lifetime of the returned Python object.

        Parameters
        ----------
        handle
            The C++ ``HostMemoryResource`` to copy from. It must have a
            back-reference installed (i.e. it must have been obtained from a
            ``BufferResource``); otherwise a ``std::bad_weak_ptr`` is raised.

        Returns
        -------
        A new Python ``HostMemoryResource`` wrapping the copied C++ handle.
        """
        cdef HostMemoryResource ret = HostMemoryResource.__new__(HostMemoryResource)
        with nogil:
            ret._handle = cpp_copy_host_mr(handle)
        return ret
