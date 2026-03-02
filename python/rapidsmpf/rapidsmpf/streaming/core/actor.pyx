# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference as deref
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

import asyncio
import inspect
from collections.abc import Awaitable, Iterable, Mapping
from functools import partial, wraps

from rapidsmpf.streaming.core.channel import Channel
from rapidsmpf.streaming.core.context import Context
from rapidsmpf.streaming.core.context cimport Context


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


class PyActor(Awaitable[None]):
    """
    A streaming actor implemented in Python.

    This runs as an Python coroutine (asyncio), which means it comes with a significant
    Python overhead. The GIL is released while C++ actors are executing.
    """
    def __init__(self, func, extra_channels, /, *args, **kwargs):
        if len(args) < 1 or not isinstance(args[0], Context):
            raise TypeError(
                "expect a Context as the first positional argument "
                "(not as a keyword argument)"
            )
        ctx = args[0]
        channels_to_shutdown = (*collect_channels(args, kwargs), *extra_channels)
        self._coro = self.run(ctx, channels_to_shutdown, func(*args, **kwargs))

    @staticmethod
    async def run(Context ctx not None, channels_to_shutdown, coro):
        """
        Run the coroutine and shutdown the channels when done.
        """
        try:
            return await coro
        finally:
            for ch in channels_to_shutdown:
                await ch.shutdown(ctx)

    def __await__(self):
        return self._coro.__await__()


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


cdef decorate_actor(extra_channels, func):
    """Validate ``func`` is async and wrap it as a PyActor."""
    if not inspect.iscoroutinefunction(func):
        raise TypeError(f"`{func.__qualname__}` must be an async function")
    return wraps(func)(partial(PyActor, func, extra_channels))


async def run_py_actors(py_actors):
    """Await all ``py_actors`` concurrently."""
    return await asyncio.gather(*py_actors)


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


def run_actor_network(*, actors, py_executor = None):
    """
    Run streaming actors to completion (blocking).

    Accepts a collection of actors. Native C++ actors are moved into the C++ network
    and executed with minimal Python overhead, while Python actors are gathered and
    executed on a dedicated event loop in the provided executor.

    Parameters
    ----------
    actors
        Iterable of actors. Each element is either a native C++ actor or a Python
        awaitable representing an actor.
    py_executor
        Executor used to run Python actors (required if any Python actors are present).
        If no Python actors are provided, this is ignored.

    Warnings
    --------
    C++ actors are released and must not be used after this call.

    Raises
    ------
    ValueError
        If Python actors are present but no executor is provided.
    Exception
        Any unhandled exception from an actor is re-raised after execution. If multiple
        actors raise exceptions, only one is re-raised, and it is unspecified which one.
    TypeError
        If actors contains an unknown actor type.

    Examples
    --------
    >>> ch: Channel[TableChunk] = context.create_channel()
    >>> cpp_actor, output = pull_from_channel(context, ch_in=ch)
    ...
    >>> @define_actor()
    ... async def python_actor(ctx: Context, ch_out: Channel) -> None:
    ...     # Send one message and close.
    ...     await ch_out.send(
    ...         context,
    ...         Message(42, TableChunk.from_pylibcudf_table(...))
    ...     )
    ...     await ch_out.drain(context)
    ...
    >>> run_actor_network(
    ...     actors=[cpp_actor, python_actor(context, ch_out=ch)],
    ...     py_executor=ThreadPoolExecutor(max_workers=1),
    ... )
    >>> results = output.release()
    >>> tbl = TableChunk.from_message(results[0])
    >>> tbl.sequence_number
    42
    """

    # Split actors into C++ actors and Python actors.
    cdef vector[cpp_Actor] cpp_actors
    cdef list py_actors = []
    for actor in actors:
        if isinstance(actor, CppActor):
            cpp_actors.push_back(move(deref((<CppActor>actor).release_handle())))
        elif isinstance(actor, PyActor):
            py_actors.append(actor)
        else:
            raise ValueError(
                "Unknown actor type, did you forget to use `@define_actor()`?"
            )

    if len(py_actors) > 0:
        if py_executor is None:
            raise ValueError("must provide a py_executor to run Python actors.")
        py_future = py_executor.submit(asyncio.run, run_py_actors(py_actors))

    try:
        if cpp_actors.size() > 0:
            with nogil:
                cpp_run_actor_network(move(cpp_actors))
    finally:
        if len(py_actors) > 0:
            py_future.result()  # This will raise any unhandled exception.
