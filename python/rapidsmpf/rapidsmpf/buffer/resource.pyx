# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference as deref
from libc.stdint cimport int64_t
from libcpp.memory cimport make_shared, shared_ptr
from libcpp.utility cimport move
from rmm.librmm.cuda_stream_pool cimport cuda_stream_pool
from rmm.pylibrmm.cuda_stream import CudaStreamFlags
from rmm.pylibrmm.cuda_stream_pool cimport CudaStreamPool
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource


cdef class MemoryReservation:
    """
    Represents a reservation for future memory allocation.

    A reservation is created by :meth:`BufferResource.reserve` and must be used
    when allocating buffers through the same :class:`BufferResource`.
    """
    def __init__(self):
        raise ValueError("use the `from_handle` factory function")

    def __dealloc__(self):
        with nogil:
            self._handle.reset()

    @staticmethod
    cdef MemoryReservation from_handle(
        unique_ptr[cpp_MemoryReservation] handle,
        BufferResource br,
    ):
        """
        Construct a MemoryReservation from an existing C++ handle.

        Parameters
        ----------
        handle
            A unique pointer to a C++ MemoryReservation.
        br
            The buffer resource associated with the reservation.

        Returns
        -------
        A new MemoryReservation wrapping the given handle.
        """
        if not handle:
            raise ValueError("handle cannot be None")
        if br is None:
            raise ValueError("br cannot be None")

        cdef MemoryReservation ret = MemoryReservation.__new__(
            MemoryReservation
        )
        ret._handle = move(handle)
        ret._br = br  # Need to keep the buffer resource alive.
        return ret

    @property
    def size(self):
        """
        Get the remaining size of the reserved memory.

        Returns
        -------
        The size of the reserved memory in bytes.
        """
        return deref(self._handle).size()

    @property
    def mem_type(self):
        """
        Get the type of memory associated with this reservation.

        Returns
        -------
        The memory type associated with this reservation.
        """
        return deref(self._handle).mem_type()

    @property
    def br(self):
        """
        Get the buffer resource associated with this reservation.

        Returns
        -------
        The buffer resource associated with this reservation.
        """
        return self._br


# Converter from `shared_ptr[cpp_LimitAvailableMemory]` to `cpp_MemoryAvailable`
cdef extern from *:
    """
    std::function<std::int64_t()> to_MemoryAvailable(
        std::shared_ptr<rapidsmpf::LimitAvailableMemory> functor
    ) {
        return *functor;
    }

    std::int64_t _call_memory_available(
        rapidsmpf::BufferResource* resource,
        rapidsmpf::MemoryType mem_type
    ) {
        return resource->memory_available(mem_type)();
    }

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
    cpp_MemoryAvailable to_MemoryAvailable(
        shared_ptr[cpp_LimitAvailableMemory]
    ) except +
    int64_t _call_memory_available(
        cpp_BufferResource* resource,
        MemoryType mem_type
    ) except + nogil
    shared_ptr[cuda_stream_pool] make_non_owning_stream_pool_ref(
        cuda_stream_pool* ptr
    ) except +


# Bindings to MemoryReservation creating methods, which we need to
# do in C++ because MemoryReservation doesn't have a default ctor.
cdef extern from * nogil:
    """
    std::pair<std::unique_ptr<rapidsmpf::MemoryReservation>, std::size_t>
    cpp_br_reserve(
        std::shared_ptr<rapidsmpf::BufferResource> br,
        rapidsmpf::MemoryType mem_type,
        size_t size,
        bool allow_overbooking
    ) {
        auto [res, ob] = br->reserve(mem_type, size, allow_overbooking);
        return {std::make_unique<rapidsmpf::MemoryReservation>(std::move(res)), ob};
    }
    """
    pair[unique_ptr[cpp_MemoryReservation], size_t] cpp_br_reserve(
        shared_ptr[cpp_BufferResource],
        MemoryType,
        size_t,
        bool_t,
    ) except +


cdef class BufferResource:
    """
    Class managing buffer resources.

    This class handles memory allocation and transfers between different memory types
    (e.g., host and device). All memory operations in RapidsMPF, such as those performed
    by the Shuffler, rely on a buffer resource for memory management.

    Parameters
    ----------
    device_mr
        Reference to the RMM device memory resource used for device allocations.
    memory_available
        Optional memory availability functions. Memory types without availability
        functions are unlimited. A function must return the current available
        memory of a specific type. It must be thread-safe if used by multiple
        `BufferResource` instances concurrently.
        Warning: calling any `BufferResource` instance methods within the function
        might result in a deadlock. This is because the buffer resource is locked
        when the function is called.
    periodic_spill_check
        Enable periodic spill checks. A dedicated thread continuously checks and
        perform spilling based on the memory availability functions. The value of
        ``periodic_spill_check`` is used as the pause between checks (in seconds).
        If None, no periodic spill check is performed.
    stream_pool
        Optional CUDA stream pool to use. If None, a new pool with 16 streams
        will be created. Must be an instance of
        ``rmm.pylibrmm.cuda_stream_pool.CudaStreamPool``.
    """
    def __cinit__(
        self,
        DeviceMemoryResource device_mr not None,
        memory_available = None,
        periodic_spill_check = 1e-3,
        stream_pool = None,
    ):
        cdef unordered_map[MemoryType, cpp_MemoryAvailable] _mem_available
        if memory_available is not None:
            for mem_type, func in memory_available.items():
                if not isinstance(func, LimitAvailableMemory):
                    raise NotImplementedError(
                        "Currently, BufferResource only accept `LimitAvailableMemory` "
                        "as memory available functions."
                    )
                _mem_available[<MemoryType?>mem_type] = to_MemoryAvailable(
                    (<LimitAvailableMemory?>func)._handle
                )
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

        # Keep MR alive because the C++ BufferResource stores a raw pointer.
        # TODO: once RMM is migrating to CCCL (copyable) any_resource,
        # rather than the any_resource_ref reference type, we don't
        # need to keep this alive here.
        self._mr = device_mr
        with nogil:
            self._handle = make_shared[cpp_BufferResource](
                device_mr.get_mr(),
                move(_mem_available),
                period,
                cpp_stream_pool,
            )
        self.spill_manager = SpillManager._create(self)

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
        cdef cpp_BufferResource* resource_ptr = self.ptr()
        # Use inline C++ to handle the function object call
        with nogil:
            ret = _call_memory_available(resource_ptr, mem_type)
        return ret

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
        OverflowError
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


cdef class LimitAvailableMemory:
    """
    A callback class for querying the remaining available memory within a defined
    limit from an RMM resource adaptor.

    This class is primarily designed to simulate constrained memory environments
    or prevent memory allocation beyond a specific threshold. It provides
    information about the available memory by subtracting the memory currently
    used (as reported by the RMM resource adaptor) from a user-defined limit.

    It is typically used in the context of memory management operations such as
    with `BufferResource`.

    Parameters
    ----------
    mr
        A statistics resource adaptor that tracks memory usage and provides
        statistics about the memory consumption. The `LimitAvailableMemory`
        instance keeps a reference to ``mr`` to keep it alive.
    limit
        The maximum memory limit (in bytes). Used to calculate the remaining
        available memory.

    Notes
    -----
    The ``mr`` resource must not be destroyed while this object is
    still in use.

    Examples
    --------
    >>> mr = RmmResourceAdaptor(...)
    >>> memory_limiter = LimitAvailableMemory(mr, limit=1_000_000)
    """
    def __init__(self, RmmResourceAdaptor mr not None, int64_t limit):
        self._mr = mr  # Keep the mr alive.
        cdef cpp_RmmResourceAdaptor* handle = mr.get_handle()
        with nogil:
            self._handle = make_shared[cpp_LimitAvailableMemory](handle, limit)

    def __call__(self):
        """
        Returns the remaining available memory within the defined limit.

        This method queries the ``rmm_statistics_resource`` to determine the memory
        currently in use and calculates the remaining memory as:
        ``limit - used_memory``.

        Returns
        -------
        int
            The remaining memory in bytes.
        """
        cdef int64_t ret
        with nogil:
            ret = deref(self._handle)()
        return ret

    def __dealloc__(self):
        with nogil:
            self._handle.reset()
