# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cpython.object cimport PyObject
from cpython.ref cimport Py_INCREF
from cython.operator cimport dereference as deref
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector

import asyncio
import inspect
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from functools import partial, wraps

from rapidsmpf._detail.exception_handling cimport ex_handler
from rapidsmpf.owning_wrapper cimport cpp_OwningWrapper
from rapidsmpf.streaming._detail.libcoro_spawn_task cimport cpp_set_py_future
from rapidsmpf.streaming.chunks.utils cimport py_deleter
from rapidsmpf.streaming.core.context cimport Context, cpp_Context

from rapidsmpf.streaming.core.channel import Channel
from rapidsmpf.streaming.core.context import Context


cdef extern from * nogil:
    """
    #include <rapidsmpf/streaming/core/coro_utils.hpp>

    namespace {
    coro::task<void> cpp_when_all_task(
        std::vector<rapidsmpf::streaming::Actor> actors
    ) {
        rapidsmpf::streaming::coro_results(co_await coro::when_all(std::move(actors)));
    }

    void cpp_when_all(
        std::shared_ptr<rapidsmpf::streaming::Context> ctx,
        std::vector<rapidsmpf::streaming::Actor> actors,
        void (*cpp_set_py_future)(void *, const char *),
        rapidsmpf::OwningWrapper py_future
    ) {
        RAPIDSMPF_EXPECTS(
            ctx->executor()->spawn_detached(
                cython_libcoro_task_wrapper(
                    cpp_set_py_future,
                    std::move(py_future),
                    cpp_when_all_task(std::move(actors))
                )
            ),
            "libcoro's spawn_detached() failed to spawn task"
        );
    }
    }
    """
    void cpp_when_all(
        shared_ptr[cpp_Context],
        vector[cpp_Actor],
        void (*cpp_set_py_future)(void*, const char *),
        cpp_OwningWrapper py_future
    ) except +ex_handler

cdef class CppActor:
    """
    A streaming actor implemented in C++.

    This represents a native C++ coroutine that runs with minimal Python
    overhead.
    """
    def __init__(self):
        raise ValueError("use the `from_handle` Cython factory function")

    @staticmethod
    cdef CppActor from_handle(unique_ptr[cpp_Actor] handle, object owner):
        """
        Create an actor from an existing native handle.

        Parameters
        ----------
        handle
            Ownership is transferred into the returned object.
        owner
            An optional Python object to keep alive for as long as this actor
            exists (e.g., to maintain resource lifetime).

        Returns
        -------
        A new actor that owns the provided handle.

        Notes
        -----
        After this call, the passed-in handle must not be used, as its
        ownership has been moved.
        """
        cdef CppActor ret = CppActor.__new__(CppActor)
        ret._handle = move(handle)
        ret._owner = owner
        return ret

    def __dealloc__(self):
        with nogil:
            self._handle.reset()

    cdef unique_ptr[cpp_Actor] release_handle(self):
        """
        Release and return the underlying native handle.

        Returns
        -------
        The native handle with ownership transferred to the caller.

        Raises
        ------
        ValueError
            If the actor is uninitialized or has already been released.

        Notes
        -----
        After calling this, the actor no longer owns a handle and should not be
        used.
        """
        if not self._handle:
            raise ValueError("CppActor is uninitialized, has it been released?")
        return move(self._handle)


def collect_channels(*objs):
    """Recursively yield all `Channel` instances found in ``objs``."""
    for obj in objs:
        if isinstance(obj, Channel):
            yield obj
        elif isinstance(obj, (str, bytes, bytearray, memoryview)):
            continue
        elif isinstance(obj, Mapping):
            yield from collect_channels(*obj.values())
        elif isinstance(obj, Iterable):
            yield from collect_channels(*obj)


async def py_actor(func, extra_channels, /, *args, **kwargs):
    """
    A streaming actor implemented in Python.

    This runs as an Python coroutine (asyncio), which means it comes with a significant
    Python overhead. The GIL is released while C++ actors are executing.
    """
    if len(args) < 1 or not isinstance(args[0], Context):
        raise TypeError(
            "expect a Context as the first positional argument "
            "(not as a keyword argument)"
        )
    ctx = args[0]
    channels_to_shutdown = (*collect_channels(args, kwargs), *extra_channels)
    try:
        return await func(*args, **kwargs)
    finally:
        for ch in channels_to_shutdown:
            await ch.shutdown(ctx)


cdef decorate_actor(extra_channels, func):
    """Validate ``func`` is async and wrap it as a PyActor."""
    if not inspect.iscoroutinefunction(func):
        raise TypeError(f"`{func.__qualname__}` must be an async function")
    return wraps(func)(partial(py_actor, func, extra_channels))


async def run_py_actors(py_actors):
    """Await all ``py_actors`` concurrently."""
    async with asyncio.TaskGroup() as tg:
        for actor in py_actors:
            tg.create_task(actor)


def define_actor(*, extra_channels=()):
    """
    Create a decorator for defining a Python streaming actor.

    The decorated coroutine must take a `Context` as its first positional argument
    and return None. When the coroutine finishes (whether successfully or with an
    exception), the wrapper automatically shuts down:
      * any channels discovered from the coroutine's arguments.
      * all channels listed in ``extra_channels``.

    Channels are discovered by recursively inspecting the coroutine's bound arguments.
    Mapping values and general iterables are traversed but byte-like objects (``str``,
    ``bytes``, ``bytearray``, ``memoryview``) are skipped.

    Parameters
    ----------
    extra_channels
        Additional channels to shut down after the decorated coroutine completes.

    Returns
    -------
    decorator
        A decorator for an async function that defines a Python actor.

    Raises
    ------
    TypeError
        If the decorated function is not async.

    Examples
    --------
    In the following example, `python_actor` is defined as a Python actor.
    When it completes, ``ch1`` is shut down automatically because it is passed
    as a coroutine argument, and ``ch2`` is shut down because it is listed in
    ``extra_channels``:
    >>> ch1: Channel[TableChunk] = context.create_channel()
    >>> ch2: Channel[TableChunk] = context.create_channel()
    ...
    >>> @define_actor(extra_channels=(ch2,))
    ... async def python_actor(ctx: Context, /, ch_in: Channel) -> None:
    ...     msg = await ch_in.recv()
    ...     await ch2.send(msg)
    ...
    ... # Calling the coroutine doesn't run it but we can provide its arguments.
    >>> actor = python_actor(context, ch_in=ch1)
    ... # Later we need to call run_actor_network() to actually run the actor.
    """

    return partial(decorate_actor, extra_channels)


def sync_wait(coro):
    """
    Run, and wait for completion of, a coroutine.

    This builds a new event loop with :class:`asyncio.Runner` and runs the
    coroutine to completion in that event loop, shutting the loop down
    afterwards.

    Notes
    -----
    This should always be called from a thread we control to ensure no live
    event loop is running.
    """
    with asyncio.Runner() as runner:
        runner.run(coro)


async def when_all(Context ctx not None, list cpp_actors):
    """
    Asynchronously run C++ actors.

    Parameters
    ----------
    ctx
        Streaming context for execution.
    cpp_actors
        List of CppActor nodes

    Warnings
    --------
    The C++ actor handles are released and must not be used after this call.
    """
    cdef vector[cpp_Actor] cpp_handles
    for actor in cpp_actors:
        cpp_handles.push_back(move(deref((<CppActor?>actor).release_handle())))
    ret = asyncio.get_running_loop().create_future()
    Py_INCREF(ret)
    with nogil:
        cpp_when_all(
            ctx._handle,
            move(cpp_handles),
            cpp_set_py_future,
            move(cpp_OwningWrapper(<void*><PyObject*>ret, py_deleter))
        )
    try:
        # This shield makes sure that if a cancellation is raised, the
        # future is not cancelled.
        await asyncio.shield(ret)
    except asyncio.CancelledError as cancel:
        # The outer awaitable was cancelled, but we must still ensure the
        # C++ future still runs to completion.
        try:
            # This could still fail so we catch and reraise the
            # cancellation but recording the C++ exception as well.
            await asyncio.shield(ret)
        except Exception as cpp_except:
            raise cancel from cpp_except
        # Otherwise just reraise the cancellation
        raise cancel


def run_actor_network(Context ctx not None, *, actors):
    """
    Run streaming actors to completion (blocking).

    Accepts a collection of actors. Native C++ actors are moved into the C++ network
    and executed with minimal Python overhead, while Python actors are gathered and
    executed on a dedicated event loop.

    Parameters
    ----------
    ctx
        Streaming context for execution.
    actors
        Iterable of actors. Each element is either a native C++ actor or a Python
        awaitable representing an actor.

    Warnings
    --------
    C++ actors are released and must not be used after this call.

    Raises
    ------
    Exception
        Any unhandled exception from an actor is re-raised after execution. If multiple
        actors raise exceptions, only one is re-raised, and it is unspecified which one.
    TypeError
        If actors contains an unknown actor type.

    Examples
    --------
    >>> ch: Channel = context.create_channel()
    >>> cpp_actor, output = pull_from_channel(context, ch_in=ch)
    ...
    >>> @define_actor()
    ... async def python_actor(ctx: Context, ch_out: Channel) -> None:
    ...     # Send one message and close.
    ...     await ch_out.send(context, Message(42, payload))
    ...     await ch_out.drain(context)
    ...
    >>> run_actor_network(
    ...     context,
    ...     actors=[cpp_actor, python_actor(context, ch_out=ch)]
    ... )
    >>> results = output.release()
    >>> results[0].sequence_number
    42
    """

    cdef list py_actors = []
    cdef list cpp_actors = []
    for actor in actors:
        if isinstance(actor, CppActor):
            cpp_actors.append(actor)
        else:
            py_actors.append(actor)

    py_actors = [when_all(ctx, cpp_actors), *py_actors]
    # Need to run in a separate thread in case the cluster runtime already
    # has an async event loop.
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(sync_wait, run_py_actors(py_actors)).result()
