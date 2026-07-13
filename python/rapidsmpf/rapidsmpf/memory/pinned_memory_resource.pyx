# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from libcpp.optional cimport optional

from rapidsmpf._detail.exception_handling cimport ex_handler


cdef extern from "<rapidsmpf/memory/pinned_memory_resource.hpp>" nogil:
    cdef bool_t cpp_is_pinned_memory_resources_supported \
        "rapidsmpf::is_pinned_memory_resources_supported"(...) except +ex_handler


cdef extern from *:
    """
    #include <optional>

    #include <rapidsmpf/memory/pinned_memory_resource.hpp>

    namespace {
    // Copy a back-referenced `PinnedMemoryResource` into an optional. The copy
    // promotes the back-reference, so the result keeps the owning
    // `BufferResource` alive. Throws `std::bad_weak_ptr` if `src` carries no
    // back-reference (which never happens for instances obtained from a
    // `BufferResource`).
    std::optional<rapidsmpf::PinnedMemoryResource>
    cpp_copy_pinned_mr(rapidsmpf::PinnedMemoryResource const& src) {
        return std::optional<rapidsmpf::PinnedMemoryResource>(src);
    }
    }  // namespace
    """
    optional[cpp_PinnedMemoryResource] cpp_copy_pinned_mr(
        const cpp_PinnedMemoryResource&
    ) except +ex_handler


cpdef bool_t is_pinned_memory_resources_supported():
    """
    Check whether pinned memory resources are supported for the current CUDA version.

    RapidsMPF requires CUDA 12.6 or newer to support pinned memory resources.
    """
    cdef bool_t ret
    with nogil:
        ret = cpp_is_pinned_memory_resources_supported()
    return ret


@dataclass
class PinnedPoolProperties:
    """
    Configuration for a pinned (page-locked) host memory pool.

    Pass an instance to
    :class:`~rapidsmpf.memory.buffer_resource.BufferResource` to enable pinned
    host memory; passing ``None`` instead disables it. The pool is only created
    when pinned host memory is supported on this system (see
    :func:`is_pinned_memory_resources_supported`).

    Attributes
    ----------
    initial_pool_size
        Initial size of the pinned host memory pool in bytes. The initial size
        is important for pinned-memory performance, especially for the first
        allocation. Defaults to ``0``.
    max_pool_size
        Maximum size of the pinned host memory pool in bytes, or ``None`` for no
        limit. Defaults to ``None``.
    numa_id
        NUMA node from which pinned host memory should be allocated, or ``None``
        to use the NUMA node of the calling thread. Defaults to ``None``.
    """
    initial_pool_size: int = 0
    max_pool_size: object = None
    numa_id: object = None


cdef object create_pinned_pool_properties_from_cpp(cpp_PinnedPoolProperties props):
    """Build a Python ``PinnedPoolProperties`` from a C++ ``PinnedPoolProperties``."""
    cdef object max_pool_size = None
    if props.max_pool_size.has_value():
        max_pool_size = props.max_pool_size.value()
    return PinnedPoolProperties(
        initial_pool_size=props.initial_pool_size,
        max_pool_size=max_pool_size,
        numa_id=props.numa_id,
    )


cdef class PinnedMemoryResource:
    """
    Opaque handle to a pinned (page-locked) host memory resource.

    The resource provides pinned host memory using a pool, enabling higher
    bandwidth and lower latency for device transfers compared to regular
    pageable host memory.

    .. rubric:: Construction

    This class cannot be constructed directly. A pinned memory resource is owned
    by a :class:`~rapidsmpf.memory.buffer_resource.BufferResource` (which installs
    the back-reference that makes the handle copyable). Configure pinned memory on
    a ``BufferResource`` and obtain the handle via
    :attr:`~rapidsmpf.memory.buffer_resource.BufferResource.pinned_mr`.

    The returned handle holds shared ownership of its owning ``BufferResource``,
    so it (and any copy of it) keeps the ``BufferResource`` alive.
    """
    def __init__(self, *args, **kwargs):
        raise TypeError(
            "PinnedMemoryResource cannot be constructed directly; configure pinned "
            "memory on a BufferResource (e.g. "
            "`BufferResource(mr, pinned_pool_properties=PinnedPoolProperties())`) and obtain it "
            "via BufferResource.pinned_mr"
        )

    def __dealloc__(self):
        with nogil:
            self._handle.reset()

    @property
    def enabled(self) -> bool:
        """
        Whether this handle wraps a valid pinned memory resource.
        """
        return self._handle.has_value()

    @staticmethod
    cdef PinnedMemoryResource _from_cpp(const cpp_PinnedMemoryResource& src):
        """
        Create a Python ``PinnedMemoryResource`` by copying a back-ref'd C++ resource.

        The copy acquires shared ownership of the owning ``BufferResource``,
        keeping it alive for the lifetime of the returned Python object.

        Parameters
        ----------
        src
            The C++ ``PinnedMemoryResource`` to copy from. Must have a
            back-reference installed (i.e. it must have been obtained from a
            ``BufferResource``); otherwise a ``std::bad_weak_ptr`` is raised.

        Returns
        -------
        A new Python ``PinnedMemoryResource`` wrapping the copied C++ resource.
        """
        cdef PinnedMemoryResource ret = PinnedMemoryResource.__new__(
            PinnedMemoryResource
        )
        ret._handle = cpp_copy_pinned_mr(src)
        return ret
