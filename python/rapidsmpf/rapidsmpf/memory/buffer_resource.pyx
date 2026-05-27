# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cython cimport no_gc_clear
from cython.operator cimport dereference as deref
from libc.stdint cimport int64_t
from libcpp cimport bool as bool_t
from libcpp.memory cimport make_shared, shared_ptr, unique_ptr
from libcpp.optional cimport optional
from libcpp.pair cimport pair
from libcpp.unordered_map cimport unordered_map
from libcpp.utility cimport move
from libcpp.vector cimport vector
from rmm.librmm.cuda_stream_pool cimport cuda_stream_pool

from rmm.pylibrmm import CudaStreamFlags

from rmm.librmm.memory_resource cimport make_any_device_resource
from rmm.pylibrmm.cuda_stream_pool cimport CudaStreamPool
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from rapidsmpf._detail.exception_handling cimport ex_handler
from rapidsmpf.memory.memory_reservation cimport MemoryReservation
from rapidsmpf.memory.pinned_memory_resource cimport (PinnedMemoryResource,
                                                      cpp_PinnedMemoryResource)
from rapidsmpf.statistics cimport Statistics


cdef extern from *:
    """
    // Helper function to create a non-owning shared_ptr from a raw pointer
    // The Python object retains ownership via its unique_ptr
    std::shared_ptr<rmm::cuda_stream_pool> make_non_owning_stream_pool_ref(
        rmm::cuda_stream_pool* ptr
    ) {
        return std::shared_ptr<rmm::cuda_stream_pool>(
            ptr, [](rmm::cuda_stream_pool*){}
        );
    }
    """
    shared_ptr[cuda_stream_pool] make_non_owning_stream_pool_ref(
        cuda_stream_pool* ptr
    ) except +ex_handler


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
        The RMM device memory resource used for device allocations. The
        BufferResource transparently wraps this resource in an internal RMM
        adaptor for allocation tracking — callers don't need to wrap it
        themselves.
    pinned_mr
        The pinned host memory resource used for :attr:`~.MemoryType.PINNED_HOST`
        allocations. If None, pinned host allocations are disabled. In that case,
        any attempt to allocate pinned memory will fail regardless of any
        ``memory_limits`` entry for ``PINNED_HOST``.
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
        The statistics instance to use. Required. Pass
        ``Statistics.disabled()`` for a no-op recorder.
    """
    def __cinit__(
        self,
        Statistics statistics not None,
        DeviceMemoryResource device_mr not None,
        *,
        PinnedMemoryResource pinned_mr = None,
        memory_limits = None,
        periodic_spill_check = 1e-3,
        stream_pool = None,
    ):
        cdef unordered_map[MemoryType, int64_t] _mem_limits
        if memory_limits is not None:
            for mem_type, limit in memory_limits.items():
                _mem_limits[<MemoryType?>mem_type] = <int64_t>limit
        cdef optional[cpp_Duration] period
        if periodic_spill_check is not None:
            period = cpp_Duration(periodic_spill_check)

        # Handle stream pool parameter
        # If None, create a default pool with 16 streams
        if stream_pool is None:
            stream_pool = CudaStreamPool(
                pool_size=16,
                flags=CudaStreamFlags.NON_BLOCKING,
            )

        if not isinstance(stream_pool, CudaStreamPool):
            raise TypeError(
                f"stream_pool must be an instance of CudaStreamPool, "
                f"got {type(stream_pool)}"
            )

        # Keep the Python stream pool alive
        self._stream_pool = stream_pool
        # Get raw pointer from the unique_ptr and create a non-owning shared_ptr
        # The Python object keeps ownership via unique_ptr, so we use a no-op deleter
        cdef shared_ptr[cuda_stream_pool] cpp_stream_pool = (
            make_non_owning_stream_pool_ref(
                (<CudaStreamPool>stream_pool).c_obj.get()
            )
        )

        # Keep statistics alive
        self._statistics = statistics
        stats_handle = statistics._handle

        # Stored for the Python device_mr/pinned_mr property accessors.
        # The C++ BufferResource owns the resource via any_resource.
        self._device_mr = device_mr
        self._pinned_mr = pinned_mr
        cdef optional[cpp_PinnedMemoryResource] cpp_pinned_mr
        if self._pinned_mr is not None:
            cpp_pinned_mr = self._pinned_mr._handle
        with nogil:
            self._handle = make_shared[cpp_BufferResource](
                stats_handle,
                make_any_device_resource(device_mr.get_mr()),
                cpp_pinned_mr,
                move(_mem_limits),
                period,
                cpp_stream_pool,
            )
        self.spill_manager = SpillManager._create(self)

    @classmethod
    def from_options(
        cls,
        Statistics statistics not None,
        DeviceMemoryResource mr not None,
        Options options not None,
    ):
        """
        Construct a BufferResource from configuration options.

        This factory method creates a BufferResource using configuration options to
        initialize all components. The supplied device memory resource is wrapped
        internally for allocation tracking — callers don't need to pre-wrap it.

        Parameters
        ----------
        statistics
            The statistics instance to use. The caller is responsible for creating and
            owning this object. Pass ``Statistics.disabled()`` for a no-op recorder.
        mr
            A device-accessible RMM memory resource.
        options
            Configuration options.

        Returns
        -------
        A BufferResource instance configured according to the options.
        """
        cdef PinnedMemoryResource pinned_mr = PinnedMemoryResource.from_options(options)
        return cls(
            statistics=statistics,
            device_mr=mr,
            pinned_mr=pinned_mr,
            memory_limits={MemoryType.DEVICE: device_limit_from_options(options)},
            periodic_spill_check=periodic_spill_check_from_options(options),
            stream_pool=stream_pool_from_options(options),
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

    cdef const cuda_stream_pool* stream_pool(self):
        return &deref(self._handle).stream_pool()

    @property
    def device_mr(self):
        """
        The memory resource used for device memory allocations.

        Returns
        -------
        The device memory resource.
        """
        return self._device_mr

    @property
    def pinned_mr(self):
        """
        The memory resource used for pinned host memory allocations.

        Returns
        -------
        The pinned host memory resource, or None if pinned host allocations
        are disabled.
        """
        return self._pinned_mr

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

    def stream_pool_size(self) -> int:
        """
        Get the size of the stream pool.

        Returns
        -------
        int
            The size of the stream pool.
        """
        return self.stream_pool().get_pool_size()

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
    cdef int pool_size = options.get_or_default("num_streams", default_value=16)
    if pool_size < 1:
        raise ValueError("the `num_streams` options must be greater than 0")
    return CudaStreamPool(
        pool_size=pool_size,
        flags=CudaStreamFlags.NON_BLOCKING,
    )
