# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cython cimport no_gc_clear
from cython.operator cimport dereference as deref
from libc.stdint cimport int64_t
from libcpp cimport bool as bool_t
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.optional cimport optional
from libcpp.pair cimport pair
from libcpp.unordered_map cimport unordered_map
from libcpp.utility cimport move
from libcpp.vector cimport vector

from rmm.pylibrmm import CudaStreamFlags

from rapidsmpf.utils.memory import check_reservation_size

from rmm.librmm.memory_resource cimport (any_resource, device_accessible,
                                         device_async_resource_ref)
from rmm.pylibrmm.cuda_stream_pool cimport CudaStreamPool
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource


cdef extern from *:
    """
    // Construct a device_async_resource_ref from an owning any_resource.
    // The free template in RMM only declares overloads for concrete RMM
    // types, this overload covers `cuda::mr::any_resource<device_accessible>`.
    std::optional<cython_device_async_resource_ref>
    cpp_make_device_async_resource_ref_from_any(
        cuda::mr::any_resource<cuda::mr::device_accessible>& mr
    ) {
        // `cython_device_async_resource_ref` is declared inline in RMM's
        // `librmm/memory_resource.pxd`.
        return std::optional<cython_device_async_resource_ref>(
            rmm::device_async_resource_ref(mr)
        );
    }
    """
    optional[device_async_resource_ref] cpp_make_device_async_resource_ref_from_any(
        any_resource[device_accessible]&
    ) except +ex_handler

from rapidsmpf._detail.exception_handling cimport ex_handler
from rapidsmpf.memory.memory_reservation cimport MemoryReservation
from rapidsmpf.memory.pinned_memory_resource cimport (
    PinnedMemoryResource, cpp_PinnedMemoryResource, cpp_PinnedPoolProperties,
    create_pinned_pool_properties_from_cpp,
    pinned_pool_properties_from_options)
from rapidsmpf.rmm_resource_adaptor cimport RmmResourceAdaptor
from rapidsmpf.statistics cimport Statistics


cdef extern from * nogil:
    """
    namespace {
    std::pair<std::unique_ptr<rapidsmpf::MemoryReservation>, std::size_t>
    cpp_br_reserve(
        std::shared_ptr<rapidsmpf::BufferResource> br,
        rapidsmpf::MemoryType mem_type,
        size_t size,
        bool allow_overbooking
    ) {
        auto ab = allow_overbooking ? rapidsmpf::AllowOverbooking::YES
                                    :rapidsmpf::AllowOverbooking::NO;
        auto [res, ob] = br->reserve(mem_type, size, ab);
        return {std::make_unique<rapidsmpf::MemoryReservation>(std::move(res)), ob};
    }

    std::unique_ptr<rapidsmpf::MemoryReservation>
    cpp_br_reserve_device_memory_and_spill(
        std::shared_ptr<rapidsmpf::BufferResource> br,
        size_t size,
        bool allow_overbooking
    ) {
        auto ab = allow_overbooking ? rapidsmpf::AllowOverbooking::YES
                                    :rapidsmpf::AllowOverbooking::NO;
        return std::make_unique<rapidsmpf::MemoryReservation>(
            br->reserve_device_memory_and_spill(size, ab)
        );
    }

    std::unique_ptr<rapidsmpf::MemoryReservation>
    cpp_br_reserve_or_fail(
        std::shared_ptr<rapidsmpf::BufferResource> br,
        size_t size,
        std::vector<rapidsmpf::MemoryType> mem_types
    ) {
        return std::make_unique<rapidsmpf::MemoryReservation>(
            br->reserve_or_fail(size, mem_types)
        );
    }
    }  // namespace
    """
    pair[unique_ptr[cpp_MemoryReservation], size_t] cpp_br_reserve(
        shared_ptr[cpp_BufferResource],
        MemoryType,
        size_t,
        bool_t,
    ) except +ex_handler
    unique_ptr[cpp_MemoryReservation] cpp_br_reserve_device_memory_and_spill(
        shared_ptr[cpp_BufferResource],
        size_t,
        bool_t,
    ) except +ex_handler
    unique_ptr[cpp_MemoryReservation] cpp_br_reserve_or_fail(
        shared_ptr[cpp_BufferResource],
        size_t,
        vector[MemoryType],
    ) except +ex_handler


@no_gc_clear
cdef class BufferResource:
    """
    Class managing buffer resources.

    This class handles memory allocation and transfers between different memory types
    (e.g., host and device). All memory operations in RapidsMPF, such as those performed
    by the Shuffler, rely on a buffer resource for memory management.

    Parameters
    ----------
    device_mr
        The RMM device memory resource used for device allocations. To ensure
        allocations are tracked for memory-limit accounting and statistics, use
        ``BufferResource.device_mr`` instead of the original ``device_mr`` after
        construction.
    pinned_pool_properties
        Configuration for the pinned host memory pool used for
        :attr:`~.MemoryType.PINNED_HOST` allocations, as a
        :class:`~rapidsmpf.memory.pinned_memory_resource.PinnedPoolProperties`.
        When ``None`` (the default), pinned host allocations are disabled and any
        attempt to allocate pinned memory will fail regardless of any
        ``memory_limits`` entry for ``PINNED_HOST``. When provided, pinned host
        memory must be supported on this system (see
        :func:`~rapidsmpf.memory.pinned_memory_resource.is_pinned_memory_resources_supported`);
        otherwise a ``RuntimeError`` is raised.
    memory_limits
        Optional mapping from :class:`~.MemoryType` to an integer byte limit.
        Memory types not present in the mapping are treated as unlimited.
    periodic_spill_check
        Enable periodic spill checks. A dedicated thread continuously checks and
        performs spilling based on memory availability. The value of
        ``periodic_spill_check`` is used as the pause between checks (in seconds).
        If None, no periodic spill check is performed.
    stream_pool
        Optional CUDA stream pool to use. If None, a new pool with 16 streams
        will be created. Must be an instance of
        ``rmm.pylibrmm.cuda_stream_pool.CudaStreamPool``.
    statistics
        The statistics instance to use. If None, a disabled statistics instance
        will be created.

    Notes
    -----
    Allocation tracking only applies to allocations routed through this
    ``BufferResource``.

    To ensure allocations are included in memory-limit accounting and
    statistics, use ``BufferResource.device_mr`` for all CUDA allocations
    associated with this resource.

    Allocations performed through other memory resources, including the
    original resource passed to the constructor or allocations made outside
    ``BufferResource``, are not tracked by this class.
    """
    def __cinit__(
        self,
        DeviceMemoryResource device_mr not None,
        *,
        pinned_pool_properties = None,
        memory_limits = None,
        periodic_spill_check = 1e-3,
        CudaStreamPool stream_pool = None,
        statistics = None,
    ):
        cdef unordered_map[MemoryType, int64_t] _mem_limits
        if memory_limits is not None:
            for mem_type, limit in memory_limits.items():
                _mem_limits[<MemoryType?>mem_type] = <int64_t>limit
        cdef optional[cpp_Duration] period
        if periodic_spill_check is not None:
            period = cpp_Duration(periodic_spill_check)

        # If None, create a default pool with 16 streams
        if stream_pool is None:
            stream_pool = CudaStreamPool(
                pool_size=16,
                flags=CudaStreamFlags.NON_BLOCKING,
            )
        self._stream_pool = stream_pool
        if statistics is None:
            statistics = Statistics(enable=False)

        # Keep statistics alive
        self._statistics = statistics
        # checked cast requires the GIL
        stats_handle = (<Statistics?>statistics)._handle

        # Keep the original Python memory resources alive while the C++
        # BufferResource holds resources derived from them. These anchors are
        # likely redundant: `any_resource[device_accessible](...)` deep-copies
        # the device MR into a self-sufficient owning any_resource, and
        # `cpp_PinnedMemoryResource` is copied by value into the C++ BR. Kept
        # defensively for now.
        # TODO: drop these once verified against pool/upstream-adaptor MRs.
        #       https://github.com/rapidsai/rapidsmpf/issues/1074
        self._device_mr = device_mr

        # The pinned resource is constructed internally by the C++
        # `BufferResource` from these properties. A None `pinned_pool_properties`
        # leaves the optional empty, disabling pinned host memory. Providing
        # properties on a system without pinned-host-memory support raises a
        # RuntimeError. A default constructed `cpp_PinnedPoolProperties` already
        # carries the C++ default NUMA node, so a `numa_id` of None keeps that default.
        cdef cpp_PinnedPoolProperties _props
        cdef optional[cpp_PinnedPoolProperties] cpp_pinned_pool
        if pinned_pool_properties is not None:
            _props.initial_pool_size = <size_t>pinned_pool_properties.initial_pool_size
            if pinned_pool_properties.max_pool_size is not None:
                _props.max_pool_size = <size_t>pinned_pool_properties.max_pool_size
            if pinned_pool_properties.numa_id is not None:
                _props.numa_id = <int>pinned_pool_properties.numa_id
            cpp_pinned_pool = _props
        with nogil:
            self._handle = cpp_BufferResource.create(
                any_resource[device_accessible](device_mr.get_mr()),
                cpp_pinned_pool,
                move(_mem_limits),
                period,
                stream_pool.c_obj,
                stats_handle,
            )
        self.spill_manager = SpillManager._create(self)

    @classmethod
    def from_options(
        cls,
        DeviceMemoryResource mr not None,
        Options options not None,
        Statistics statistics=None,
    ):
        """
        Construct a BufferResource from configuration options.

        This factory method creates a BufferResource using configuration options to
        initialize all components. The supplied device memory resource is wrapped
        internally for allocation tracking — callers don't need to pre-wrap it.

        Parameters
        ----------
        mr
            A device-accessible RMM memory resource.
        options
            Configuration options.
        statistics
            The statistics instance to use. The caller is responsible for creating and
            owning this object. Defaults to ``Statistics.disabled()``.

        Returns
        -------
        A BufferResource instance configured according to the options.
        """
        if statistics is None:
            statistics = Statistics.disabled()

        # Derive the pinned pool configuration from the options; an empty optional
        # means pinned host memory is disabled.
        cdef optional[cpp_PinnedPoolProperties] props = \
            pinned_pool_properties_from_options(options._handle)
        pinned_pool_properties = None
        if props.has_value():
            pinned_pool_properties = create_pinned_pool_properties_from_cpp(
                props.value()
            )

        return cls(
            device_mr=mr,
            pinned_pool_properties=pinned_pool_properties,
            memory_limits={MemoryType.DEVICE: device_limit_from_options(options)},
            periodic_spill_check=periodic_spill_check_from_options(options),
            stream_pool=stream_pool_from_options(options),
            statistics=statistics,
        )

    def __dealloc__(self):
        """
        Deallocate resource without holding the GIL.

        This is important to ensure owned resources, like the underlying C++
        `SpillManager` object is destroyed, ensuring any threads can be
        joined without risk of deadlocks if both thread compete for the GIL.
        """
        with nogil:
            self._handle.reset()

    cdef cpp_BufferResource* ptr(self):
        """
        A raw pointer to the underlying C++ `BufferResource`.

        Returns
        -------
            The raw pointer.
        """
        return self._handle.get()

    @property
    def stream_pool(self):
        """
        The stream pool associated with this buffer resource.

        Returns
        -------
        An RMM CudaStreamPool.
        """
        return self._stream_pool

    @property
    def device_mr(self):
        """
        The tracked device memory resource.

        Allocations made through this resource count against the ``BufferResource``
        memory limits and appear in its statistics.

        Returns
        -------
        The tracked device memory resource.
        """
        return OwningDeviceMemoryResource._create(
            any_resource[device_accessible](deref(self._handle).device_mr())
        )

    cpdef RmmResourceAdaptor device_mr_adaptor(self):
        """
        The internal device memory resource adaptor with a back-reference installed.

        Returns a copyable ``RmmResourceAdaptor`` that holds shared ownership of this
        ``BufferResource``, keeping it alive for as long as the returned adaptor (or
        any copies of it) lives.

        This is the only way to obtain an ``RmmResourceAdaptor``; use it when you need
        to pass one to APIs that copy the adaptor, such as
        :meth:`~rapidsmpf.statistics.Statistics.memory_profiling` or
        :meth:`~rapidsmpf.statistics.Statistics.report`.

        Returns
        -------
        A back-ref'd ``RmmResourceAdaptor`` whose copies keep this ``BufferResource``
        alive.
        """
        return RmmResourceAdaptor._from_cpp(deref(self._handle).device_mr_adaptor())

    @property
    def pinned_mr(self):
        """
        The memory resource used for pinned host memory allocations.

        The returned handle holds shared ownership of this ``BufferResource``,
        keeping it alive for as long as the handle (or any copy of it) lives.

        Returns
        -------
        The pinned host memory resource, or None if pinned host allocations
        are disabled.
        """
        cdef optional[cpp_PinnedMemoryResource] opt
        with nogil:
            opt = deref(self._handle).try_pinned_mr()
        if not opt.has_value():
            return None
        return PinnedMemoryResource.from_handle(opt)

    def memory_reserved(self, MemoryType mem_type):
        """
        Get the current reserved memory of the specified memory type.

        Parameters
        ----------
        mem_type
            The target memory type.

        Returns
        -------
        The memory reserved, in bytes.
        """
        cdef size_t ret
        with nogil:
            ret = deref(self._handle).memory_reserved(mem_type)
        return ret

    def memory_available(self, MemoryType mem_type):
        """
        Get the current available memory of the specified memory type.
        """
        cdef int64_t ret
        with nogil:
            ret = deref(self._handle).memory_available(mem_type)
        return ret

    def set_memory_limit(self, MemoryType mem_type, int64_t limit):
        """
        Set the byte limit for the specified memory type.

        The store is atomic, but readers (e.g. ``memory_available()`` and
        ``reserve()``) observe the limit and the allocation count independently.
        A concurrent ``set_memory_limit()`` call can change the limit between a
        caller's read of ``memory_available()`` and a subsequent allocation
        decision; callers that need a coherent view must serialize updates with
        higher-level synchronization.

        Parameters
        ----------
        mem_type
            The memory type whose limit is being updated.
        limit
            The new byte limit. Negative values are allowed; they make
            ``memory_available(mem_type)`` always negative and so trigger
            continuous spilling.
        """
        with nogil:
            deref(self._handle).set_memory_limit(mem_type, limit)

    def reserve(self, MemoryType mem_type, size_t size, *, bool_t allow_overbooking):
        """
        Reserve an amount of the specified memory type.

        Creates a new reservation of the specified size and memory type to inform the
        system about upcoming buffer allocations.

        If overbooking is allowed, a reservation of the requested `size` is returned
        even if the memory is not currently available. In that case, the caller must
        guarantee that at least the overbooked amount of memory will be freed before
        the reservation is used.

        If overbooking is not allowed, a reservation of size zero is returned on
        failure.

        Parameters
        ----------
        mem_type
            The target memory type.
        size
            The number of bytes to reserve.
        allow_overbooking
            Whether overbooking is permitted.

        Returns
        -------
        A tuple (reservation, overbooked_bytes):
            - On success, the reservation's size equals `size`.
            - On failure, the reservation's size equals zero (a zero-sized reservation
              never fails).
        """
        check_reservation_size(size)
        cdef pair[unique_ptr[cpp_MemoryReservation], size_t] ret
        with nogil:
            ret = cpp_br_reserve(self._handle, mem_type, size, allow_overbooking)
        return MemoryReservation.from_handle(move(ret.first), self), ret.second

    def reserve_device_memory_and_spill(
        self, size_t size, *, bool_t allow_overbooking
    ):
        """
        Reserve device memory and spill if necessary.

        Attempts to reserve the requested amount of device memory. If insufficient
        memory is available, spilling is triggered to free space. When overbooking
        is allowed, the reservation may succeed even if spilling was not sufficient
        to fully satisfy the request.

        Parameters
        ----------
        size
            The amount of memory to reserve.
        allow_overbooking
            Whether to allow overbooking. If false, ensures enough memory is freed
            to satisfy the reservation. If true, the reservation may succeed even
            if spilling was insufficient.

        Returns
        -------
        The resulting memory reservation.

        Raises
        ------
        ReservationError
            If overbooking is disabled and the buffer resource cannot free enough
            device memory through spilling to satisfy the request.
        """
        check_reservation_size(size)
        cdef unique_ptr[cpp_MemoryReservation] ret
        with nogil:
            ret = cpp_br_reserve_device_memory_and_spill(
                self._handle, size, allow_overbooking
            )
        return MemoryReservation.from_handle(move(ret), self)

    def reserve_or_fail(self, size_t size, list mem_types):
        """
        Make a memory reservation or fail based on the given order of memory types.

        Attempts to reserve memory by iterating over ``mem_types`` in the given order
        of preference. For each memory type, a reservation without overbooking is
        requested. If no memory type can satisfy the request, a ``RuntimeError`` is
        raised.

        Parameters
        ----------
        size
            The number of bytes to reserve.
        mem_types
            List of :class:`~.MemoryType` values specifying the order of preference
            in which memory types are tried.

        Returns
        -------
        A :class:`~.MemoryReservation` for the first memory type that could satisfy
        the request.

        Raises
        ------
        RuntimeError
            If no memory type in ``mem_types`` could satisfy the reservation.
        """
        check_reservation_size(size)
        cdef vector[MemoryType] cpp_mem_types
        for mt in mem_types:
            cpp_mem_types.push_back(<MemoryType?>mt)
        cdef unique_ptr[cpp_MemoryReservation] ret
        with nogil:
            ret = cpp_br_reserve_or_fail(self._handle, size, cpp_mem_types)
        return MemoryReservation.from_handle(move(ret), self)

    def release(self, MemoryReservation reservation not None, size_t size):
        """
        Consume a portion of the reserved memory.

        Reduces the remaining size of the reserved memory by the specified amount.

        Parameters
        ----------
        reservation
            The memory reservation to consume from.
        size
            The number of bytes to consume.

        Returns
        -------
        The remaining size of the reserved memory after consumption.

        Raises
        ------
        ReservationError
            If the released size exceeds the total reserved size.
        """
        cdef size_t ret
        with nogil:
            ret = deref(self._handle).release(deref(reservation._handle), size)
        return ret

    @property
    def statistics(self):
        """
        Gets the statistics instance associated with this buffer resource.

        Returns
        -------
        The Statistics instance.
        """
        return self._statistics


cdef extern from "<rapidsmpf/memory/buffer_resource.hpp>" nogil:
    cdef int64_t cpp_device_limit_from_options \
        "rapidsmpf::device_limit_from_options"(
            cpp_Options options
        ) except +ex_handler

    cdef optional[cpp_Duration] cpp_periodic_spill_check_from_options \
        "rapidsmpf::periodic_spill_check_from_options"(
            cpp_Options options
        ) except +ex_handler


def device_limit_from_options(Options options not None):
    """
    Get the ``spill_device_limit`` parameter from configuration options.

    Reads the ``spill_device_limit`` option, falling back to 80% of total device
    memory when unset.

    Parameters
    ----------
    options
        Configuration options.

    Returns
    -------
    int
        The device memory limit in bytes.
    """
    cdef int64_t ret
    with nogil:
        ret = cpp_device_limit_from_options(options._handle)
    return ret


def periodic_spill_check_from_options(Options options not None):
    """
    Get the ``periodic_spill_check`` parameter from configuration options.

    Parameters
    ----------
    options
        Configuration options.

    Returns
    -------
    The duration of the pause between spill checks in seconds, or ``None`` if
    periodic spill checks are disabled.
    """
    cdef optional[cpp_Duration] ret
    with nogil:
        ret = cpp_periodic_spill_check_from_options(options._handle)
    if not ret.has_value():
        return None
    return ret.value().count()


def stream_pool_from_options(Options options not None):
    """
    Create a new CUDA stream pool from configuration options.

    Parameters
    ----------
    options
        Configuration options.

    Returns
    -------
    Pool of CUDA streams used throughout RapidsMPF for operations that do not take
    an explicit CUDA stream.
    """
    cdef int pool_size = options.get(
        "num_streams",
        return_type=int,
        factory=int,
    )
    if pool_size < 1:
        raise ValueError(
            "the `num_streams` options must be greater than 0"
        )
    return CudaStreamPool(
        pool_size=pool_size,
        flags=CudaStreamFlags.NON_BLOCKING,
    )


cdef class OwningDeviceMemoryResource(DeviceMemoryResource):
    """
    Owning ``DeviceMemoryResource``.

    Useful for exposing device memory resources to Python in a form that is
    compatible with cuDF/RMM APIs while preserving ownership semantics.

    Notes
    -----
    RMM does not currently provide an equivalent owning wrapper. If one is
    added in the future, this class can likely be replaced by the
    RMM-provided implementation.
    """
    @staticmethod
    cdef OwningDeviceMemoryResource _create(
        any_resource[device_accessible] resource,
    ):
        cdef OwningDeviceMemoryResource self = (
            OwningDeviceMemoryResource.__new__(OwningDeviceMemoryResource)
        )
        # Owning storage for the underlying CCCL resource. The base class's
        # `c_ref` is a non-owning view into `c_obj`.
        self.c_obj = resource
        self.c_ref = cpp_make_device_async_resource_ref_from_any(self.c_obj)
        return self
