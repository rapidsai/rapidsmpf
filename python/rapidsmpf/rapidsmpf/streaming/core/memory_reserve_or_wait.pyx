# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cpython.object cimport PyObject
from cpython.ref cimport Py_INCREF
from cython.operator cimport dereference as deref
from libc.stdint cimport int64_t
from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.utility cimport move

from rapidsmpf.config cimport Options, cpp_Options
from rapidsmpf.memory.buffer cimport MemoryType
from rapidsmpf.memory.memory_reservation cimport (MemoryReservation,
                                                  cpp_MemoryReservation)
from rapidsmpf.owning_wrapper cimport cpp_OwningWrapper
from rapidsmpf.streaming._detail.libcoro_spawn_task cimport cpp_set_py_future
from rapidsmpf.streaming.chunks.utils cimport py_deleter
from rapidsmpf.streaming.core.context cimport Context, cpp_Context

import asyncio


cdef extern from * nogil:
    """
    namespace {
    auto cpp_make_shared(
        rapidsmpf::config::Options options,
        rapidsmpf::MemoryType mem_type,
        rapidsmpf::streaming::Context &ctx
    ) {
        return std::make_shared<rapidsmpf::streaming::MemoryReserveOrWait>(
            std::move(options), mem_type, ctx.executor(), ctx.br()
        );
    }
    }  // namespace
    """
    shared_ptr[cpp_MemoryReserveOrWait] cpp_make_shared(
        cpp_Options options,
        MemoryType mem_type,
        cpp_Context &ctx
    ) except +

cdef extern from * nogil:
    """
    namespace {
    coro::task<void> shutdown_task(
        std::shared_ptr<rapidsmpf::streaming::MemoryReserveOrWait> mrow
    ) {
        co_await mrow->shutdown();
    }

    void cpp_shutdown(
        std::shared_ptr<rapidsmpf::streaming::MemoryReserveOrWait> mrow,
        void (*cpp_set_py_future)(void*, const char *),
        rapidsmpf::OwningWrapper py_future
    ) {
        RAPIDSMPF_EXPECTS(
            mrow->executor()->spawn_detached(
                cython_libcoro_task_wrapper(
                    cpp_set_py_future,
                    std::move(py_future),
                    shutdown_task(std::move(mrow))
                )
            ),
            "libcoro's spawn_detached() failed to spawn task"
        );
    }
    }  // namespace
    """
    void cpp_shutdown(
        shared_ptr[cpp_MemoryReserveOrWait] mrow,
        void (*cpp_set_py_future)(void*, const char *),
        cpp_OwningWrapper py_future
    ) except +

cdef extern from * nogil:
    """
    namespace {
    coro::task<void> reserve_or_wait_task(
        std::shared_ptr<rapidsmpf::streaming::MemoryReserveOrWait> mrow,
        std::size_t size,
        std::size_t net_memory_delta,
        std::shared_ptr<std::unique_ptr<rapidsmpf::MemoryReservation>> output
    ) {
        *output = std::make_unique<rapidsmpf::MemoryReservation>(
            co_await mrow->reserve_or_wait(size, net_memory_delta)
        );
    }

    auto cpp_reserve_or_wait(
        std::shared_ptr<rapidsmpf::streaming::MemoryReserveOrWait> mrow,
        std::size_t size,
        std::size_t net_memory_delta,
        void (*cpp_set_py_future)(void*, const char *),
        rapidsmpf::OwningWrapper py_future
    ) {
        auto output = std::make_shared<std::unique_ptr<rapidsmpf::MemoryReservation>>();
        RAPIDSMPF_EXPECTS(
            mrow->executor()->spawn_detached(
                cython_libcoro_task_wrapper(
                    cpp_set_py_future,
                    std::move(py_future),
                    reserve_or_wait_task(
                        std::move(mrow),
                        size,
                        net_memory_delta,
                        output
                    )
                )
            ),
            "libcoro's spawn_detached() failed to spawn task"
        );
        return output;
    }
    }  // namespace
    """
    shared_ptr[unique_ptr[cpp_MemoryReservation]] cpp_reserve_or_wait(
        shared_ptr[cpp_MemoryReserveOrWait] mrow,
        size_t size,
        size_t net_memory_delta,
        void (*cpp_set_py_future)(void*, const char *),
        cpp_OwningWrapper py_future
    ) except +

cdef extern from * nogil:
    """
    namespace {
    coro::task<void> reserve_or_wait_or_overbook_task(
        std::shared_ptr<rapidsmpf::streaming::MemoryReserveOrWait> mrow,
        std::size_t size,
        std::size_t net_memory_delta,
        std::shared_ptr<
            std::pair<std::unique_ptr<rapidsmpf::MemoryReservation>, std::size_t>
        > output
    ) {
        auto [res, overbooking] = co_await mrow->reserve_or_wait_or_overbook(
            size, net_memory_delta
        );
        *output = {
            std::make_unique<rapidsmpf::MemoryReservation>(std::move(res)),
            overbooking
        };
    }

    auto cpp_reserve_or_wait_or_overbook(
        std::shared_ptr<rapidsmpf::streaming::MemoryReserveOrWait> mrow,
        std::size_t size,
        std::size_t net_memory_delta,
        void (*cpp_set_py_future)(void*, const char *),
        rapidsmpf::OwningWrapper py_future
    ) {
        auto output = std::make_shared<
            std::pair<std::unique_ptr<rapidsmpf::MemoryReservation>, std::size_t>
        >();
        RAPIDSMPF_EXPECTS(
            mrow->executor()->spawn_detached(
                cython_libcoro_task_wrapper(
                    cpp_set_py_future,
                    std::move(py_future),
                    reserve_or_wait_or_overbook_task(
                        std::move(mrow),
                        size,
                        net_memory_delta,
                        output
                    )
                )
            ),
            "libcoro's spawn_detached() failed to spawn task"
        );
        return output;
    }
    }  // namespace
    """
    shared_ptr[pair[unique_ptr[cpp_MemoryReservation], size_t]] \
        cpp_reserve_or_wait_or_overbook(
        shared_ptr[cpp_MemoryReserveOrWait] mrow,
        size_t size,
        size_t net_memory_delta,
        void (*cpp_set_py_future)(void*, const char *),
        cpp_OwningWrapper py_future
    ) except +

cdef extern from * nogil:
    """
    namespace {
    coro::task<void> reserve_or_wait_or_fail_task(
        std::shared_ptr<rapidsmpf::streaming::MemoryReserveOrWait> mrow,
        std::size_t size,
        std::size_t net_memory_delta,
        std::shared_ptr<std::unique_ptr<rapidsmpf::MemoryReservation>> output
    ) {
        *output = std::make_unique<rapidsmpf::MemoryReservation>(
            co_await mrow->reserve_or_wait_or_fail(size, net_memory_delta)
        );
    }

    auto cpp_reserve_or_wait_or_fail(
        std::shared_ptr<rapidsmpf::streaming::MemoryReserveOrWait> mrow,
        std::size_t size,
        std::size_t net_memory_delta,
        void (*cpp_set_py_future)(void*, const char *),
        rapidsmpf::OwningWrapper py_future
    ) {
        auto output = std::make_shared<std::unique_ptr<rapidsmpf::MemoryReservation>>();
        RAPIDSMPF_EXPECTS(
            mrow->executor()->spawn_detached(
                cython_libcoro_task_wrapper(
                    cpp_set_py_future,
                    std::move(py_future),
                    reserve_or_wait_or_fail_task(
                        std::move(mrow),
                        size,
                        net_memory_delta,
                        output
                    )
                )
            ),
            "libcoro's spawn_detached() failed to spawn task"
        );
        return output;
    }
    }  // namespace
    """
    shared_ptr[unique_ptr[cpp_MemoryReservation]] cpp_reserve_or_wait_or_fail(
        shared_ptr[cpp_MemoryReserveOrWait] mrow,
        size_t size,
        size_t net_memory_delta,
        void (*cpp_set_py_future)(void*, const char *),
        cpp_OwningWrapper py_future
    ) except +


cdef class MemoryReserveOrWait:
    """
    Asynchronous coordinator for memory reservation requests.

    ``MemoryReserveOrWait`` provides a coroutine-based mechanism for reserving memory
    with backpressure. Callers submit reservation requests via :meth:`reserve_or_wait`,
    which suspends until sufficient memory becomes available or progress must be forced.

    A background task is spawned on demand to periodically check available memory and
    fulfill pending requests. If no reservation request can be satisfied within the
    timeout specified by the ``"memory_reserve_timeout_ms"`` option, the scheduler
    forces progress by selecting the smallest pending request and attempting to reserve
    memory for it. This attempt may result in an empty reservation if the request still
    cannot be satisfied.

    The timeout provides a global progress guarantee and does not apply to a specific
    reservation request. Instead, it bounds how long the system may go without
    satisfying any pending request.

    Parameters
    ----------
    options
        Configuration options. The option ``"memory_reserve_timeout_ms"`` controls the
        global progress timeout and defaults to 100 ms.
    mem_type
        The memory type for which reservations are requested.
    ctx
        Node context used during construction to read context properties. The context
        is not kept alive after initialization.

    Raises
    ------
    RuntimeError
        If shutdown occurs before a reservation request can be processed.
    """
    def __init__(
        self, Options options not None, MemoryType mem_type, Context ctx not None
    ):
        self._br = ctx.br()
        with nogil:
            self._handle = cpp_make_shared(
                options._handle,
                mem_type,
                deref(ctx._handle),
            )

    @staticmethod
    cdef MemoryReserveOrWait from_handle(
        shared_ptr[cpp_MemoryReserveOrWait] handle, BufferResource br
    ):
        """
        Construct a MemoryReserveOrWait from an existing C++ handle.

        Parameters
        ----------
        handle
            A shared pointer to a C++ MemoryReserveOrWait.
        br
            The associated buffer resource.

        Returns
        -------
        A new MemoryReserveOrWait wrapping the given handle.
        """
        assert br is not None
        cdef MemoryReserveOrWait ret = MemoryReserveOrWait.__new__(MemoryReserveOrWait)
        ret._handle = handle
        ret._br = br
        return ret

    def __dealloc__(self):
        with nogil:
            self._handle.reset()

    async def shutdown(self):
        """
        Shut down all pending memory reservation requests.

        Cancels all pending reservation requests and signals the background
        periodic memory check task to exit. The returned coroutine completes
        only after all pending requests have been cancelled and the periodic
        memory check task has fully exited.

        Returns
        -------
        A coroutine that completes once shutdown is complete.
        """
        ret = asyncio.get_running_loop().create_future()
        Py_INCREF(ret)
        with nogil:
            cpp_shutdown(
                self._handle,
                cpp_set_py_future,
                move(cpp_OwningWrapper(<void*><PyObject*>ret, py_deleter)),
            )
        await ret

    async def reserve_or_wait(self, size_t size, *, int64_t net_memory_delta):
        """
        Attempt to reserve memory, or wait until progress can be made.

        Submits a memory reservation request and suspends until either sufficient
        memory becomes available or no reservation request, including other pending
        requests, makes progress within the configured timeout.

        The timeout does not apply specifically to this request. Instead, it serves as
        a global progress guarantee. If no pending reservation request can be satisfied
        within the timeout, ``MemoryReserveOrWait`` forces progress by selecting the
        smallest pending request and attempting to reserve memory for it. The forced
        reservation attempt may result in an empty :class:`MemoryReservation` if the
        selected request still cannot be satisfied.

        When multiple reservation requests are eligible, ``MemoryReserveOrWait`` uses
        ``net_memory_delta`` as a heuristic to prefer requests that are expected to
        reduce memory pressure sooner. The value represents the estimated net change in
        memory usage after the reservation has been granted and the dependent operation
        completes (that is, after both reserving ``size`` bytes and completing the work
        that consumes the reservation):
            - > 0: expected net increase in memory usage
            - = 0: memory-neutral
            - < 0: expected net decrease in memory usage

        Smaller values have higher priority.

        Examples
        --------
        Reading data from disk into memory typically has a positive ``net_memory_delta``
        because memory usage increases.

        A row-wise transformation that retains input and output typically has a
        ``net_memory_delta`` near zero.

        Writing data to disk or a reduction that frees inputs typically has a negative
        ``net_memory_delta`` because memory usage decreases.

        Parameters
        ----------
        size
            Number of bytes to reserve.
        net_memory_delta
            Estimated net change in memory usage after the reservation has been granted
            and the dependent operation completes. Smaller values have higher priority.

        Returns
        -------
        A memory reservation representing the allocated memory. The reservation may be
        empty if progress could not be made.

        Raises
        ------
        RuntimeError
            If shutdown occurs before the request can be processed.
        """
        cdef shared_ptr[unique_ptr[cpp_MemoryReservation]] c_ret
        future = asyncio.get_running_loop().create_future()
        Py_INCREF(future)
        with nogil:
            c_ret = cpp_reserve_or_wait(
                self._handle,
                size,
                net_memory_delta,
                cpp_set_py_future,
                move(cpp_OwningWrapper(<void*><PyObject*>future, py_deleter))
            )
        await future
        if not c_ret:
            assert False, "something went wrong, task returned a null pointer!"
        return MemoryReservation.from_handle(move(deref(c_ret)), self._br)

    async def reserve_or_wait_or_overbook(
        self, size_t size, *, int64_t net_memory_delta
    ):
        """
        Variant of `reserve_or_wait()` that allows overbooking on timeout.

        This coroutine behaves identically to `reserve_or_wait()` with respect to
        request submission, waiting, and progress guarantees. The only difference is
        the behavior when the progress timeout expires.

        If no reservation request can be satisfied before the timeout, this method
        attempts to reserve the requested memory by allowing overbooking. This
        guarantees forward progress, but may exceed the configured memory limits.

        Parameters
        ----------
        size
            Number of bytes to reserve.
        net_memory_delta
            Heuristic used to prioritize eligible requests. See `reserve_or_wait()`
            for details and semantics.

        Returns
        -------
        A pair consisting of:
            - A `MemoryReservation` representing the allocated memory.
            - The number of bytes by which the reservation overbooked the available
              memory. This value is zero if no overbooking occurred.

        Raises
        ------
        RuntimeError
            If shutdown occurs before the request can be processed.
        """
        cdef shared_ptr[pair[unique_ptr[cpp_MemoryReservation], size_t]] c_ret
        future = asyncio.get_running_loop().create_future()
        Py_INCREF(future)
        with nogil:
            c_ret = cpp_reserve_or_wait_or_overbook(
                self._handle,
                size,
                net_memory_delta,
                cpp_set_py_future,
                move(cpp_OwningWrapper(<void*><PyObject*>future, py_deleter)),
            )
        await future
        if not c_ret:
            assert False, "something went wrong, task returned a null pointer!"
        return (
            MemoryReservation.from_handle(move(deref(c_ret).first), self._br),
            deref(c_ret).second
        )

    async def reserve_or_wait_or_fail(self, size_t size, *, int64_t net_memory_delta):
        """
        Variant of `reserve_or_wait()` that fails if no progress is possible.

        This coroutine behaves identically to `reserve_or_wait()` with respect to
        request submission, waiting, and progress guarantees until the progress
        timeout expires.

        If no reservation request can be satisfied before the timeout, this method
        fails instead of forcing progress. Overbooking is not allowed, and no memory
        reservation is made.

        Parameters
        ----------
        size
            Number of bytes to reserve.
        net_memory_delta
            Heuristic used to prioritize eligible requests. See `reserve_or_wait()`
            for details and semantics.

        Returns
        -------
        The memory reservation representing the allocated memory.

        Raises
        ------
        RuntimeError
            If no progress is possible within the timeout.
        RuntimeError
            If shutdown occurs before the request can be processed.

        See Also
        --------
        reserve_or_wait
        """
        cdef shared_ptr[unique_ptr[cpp_MemoryReservation]] c_ret
        future = asyncio.get_running_loop().create_future()
        Py_INCREF(future)
        with nogil:
            c_ret = cpp_reserve_or_wait_or_fail(
                self._handle,
                size,
                net_memory_delta,
                cpp_set_py_future,
                move(cpp_OwningWrapper(<void*><PyObject*>future, py_deleter))
            )
        await future
        if not c_ret:
            assert False, "something went wrong, task returned a null pointer!"
        return MemoryReservation.from_handle(move(deref(c_ret)), self._br)

    def size(self):
        """
        Return the number of pending memory reservation requests.

        The returned value is a snapshot and may change concurrently as
        reservation requests are added or fulfilled.

        Returns
        -------
        The number of outstanding reservation requests.
        """
        cdef size_t _ret
        with nogil:
            _ret = deref(self._handle).size()
        return _ret


async def reserve_memory(
    Context ctx not None,
    size,
    *,
    net_memory_delta,
    mem_type = MemoryType.DEVICE,
    allow_overbooking = None
):
    """
    Reserve memory using the context memory reservation mechanism.

    Submits a memory reservation request for the specified memory type and
    suspends until the request is satisfied or no further progress can be
    made. The behavior when the progress timeout expires depends on whether
    overbooking is allowed.

    This is a convenience helper that returns only the memory reservation.
    If more control is required, for example inspecting the amount of
    overbooking, callers should use the context memory reservation system
    directly, such as
    ``ctx.memory(MemoryType.DEVICE).reserve_or_wait_or_overbook(...)``.

    Parameters
    ----------
    ctx
        Node context used to obtain the memory reservation handle.
    size
        Number of bytes to reserve.
    net_memory_delta
        Heuristic used to prioritize eligible requests. See
        `MemoryReserveOrWait.reserve_or_wait()` for details and semantics.
    mem_type
        Memory type for which to reserve memory.
    allow_overbooking
        Whether to allow overbooking if no progress is possible.
          - If ``True``, the reservation may overbook memory when no further
            progress can be made. If ``False``, the call fails when no progress
            is possible.
          - If ``None`` (the default), the behavior is determined by the
            configuration option ``"allow_overbooking_by_default"``, which is
            read via ``ctx.options()``.

    Returns
    -------
    The allocated memory reservation.

    Raises
    ------
    RuntimeError
        If shutdown occurs before the request can be processed.
    RuntimeError
        If no further progress is possible and overbooking is disabled.

    Examples
    --------
    Reserve device memory inside a node:
    >>> res = await reserve_memory(
    ...     ctx,
    ...     size=1024,
    ...     net_memory_delta=0,
    ...     allow_overbooking=True,
    ... )
    >>> res.size
    1024

    Disable overbooking and fail if no progress is possible:
    >>> res = await reserve_memory(
    ...     ctx,
    ...     size=2048,
    ...     net_memory_delta=0,
    ...     allow_overbooking=False,
    ... )
    """
    if allow_overbooking is None:
        allow_overbooking = ctx.options().get_or_default(
            "allow_overbooking_by_default", default_value=True
        )

    memory = ctx.memory(mem_type)
    if allow_overbooking:
        ret, _ = await memory.reserve_or_wait_or_overbook(
            size=size, net_memory_delta=net_memory_delta
        )
    else:
        ret = await memory.reserve_or_wait_or_fail(
            size=size, net_memory_delta=net_memory_delta
        )
    return ret
