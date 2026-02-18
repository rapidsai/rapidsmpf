# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference as deref
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

import asyncio
import inspect
from collections.abc import Awaitable, Iterable, Iterator, Mapping
from functools import wraps

from rapidsmpf.streaming.core.channel import Channel
from rapidsmpf.streaming.core.context import Context


cdef class CppNode:
    """
    A streaming node implemented in C++.

    This represents a native C++ coroutine that runs with minimal Python
    overhead.
    """
    def __init__(self):
        raise ValueError("use the `from_handle` Cython factory function")

    @staticmethod
    cdef CppNode from_handle(unique_ptr[cpp_Node] handle, object owner):
        """
        Create a node from an existing native handle.

        Parameters
        ----------
        handle
            Ownership is transferred into the returned object.
        owner
            An optional Python object to keep alive for as long as this node
            exists (e.g., to maintain resource lifetime).

        Returns
        -------
        A new node that owns the provided handle.

        Notes
        -----
        After this call, the passed-in handle must not be used, as its
        ownership has been moved.
        """
        cdef CppNode ret = CppNode.__new__(CppNode)
        ret._handle = move(handle)
        ret._owner = owner
        return ret

    def __dealloc__(self):
        with nogil:
            self._handle.reset()

    cdef unique_ptr[cpp_Node] release_handle(self):
        """
        Release and return the underlying native handle.

        Returns
        -------
        The native handle with ownership transferred to the caller.

        Raises
        ------
        ValueError
            If the node is uninitialized or has already been released.

        Notes
        -----
        After calling this, the node no longer owns a handle and should not be
        used.
        """
        if not self._handle:
            raise ValueError("CppNode is uninitialized, has it been released?")
        return move(self._handle)


class PyNode(Awaitable[None]):
    """
    A streaming node implemented in Python.

    This runs as an Python coroutine (asyncio), which means it comes with a significant
    Python overhead. The GIL is released on `await` and when calling the C++ API.
    """
    def __init__(self, coro: Awaitable[None]) -> None:
        self._coro = coro

    def __await__(self) -> Iterator[None]:
        return self._coro.__await__()


def define_py_node(*, extra_channels=()):
    """
    Create a decorator for defining a Python streaming node.

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
        A decorator for an async function that defines a Python node.

    Raises
    ------
    TypeError
        If the decorated function is not async.

    Examples
    --------
    In the following example, `python_node` is defined as a Python node.
    When it completes, ``ch1`` is shut down automatically because it is passed
    as a coroutine argument, and ``ch2`` is shut down because it is listed in
    ``extra_channels``:
    >>> ch1: Channel[TableChunk] = context.create_channel()
    >>> ch2: Channel[TableChunk] = context.create_channel()
    ...
    >>> @define_py_node(extra_channels=(ch2,))
    ... async def python_node(ctx: Context, /, ch_in: Channel) -> None:
    ...     msg = await ch_in.recv()
    ...     await ch2.send(msg)
    ...
    ... # Calling the coroutine doesn't run it but we can provide its arguments.
    >>> node = python_node(context, ch_in=ch1)
    ... # Later we need to call run_streaming_pipeline() to actually run the node.
    """

    def _collect_channels(obj, out):
        if isinstance(obj, Channel):
            out.append(obj)
        elif isinstance(obj, (str, bytes, bytearray, memoryview)):
            return
        elif isinstance(obj, Mapping):
            for v in obj.values():
                _collect_channels(v, out)
        elif isinstance(obj, Iterable):
            for v in obj:
                _collect_channels(v, out)

    def decorator(func):
        if not inspect.iscoroutinefunction(func):
            raise TypeError(f"`{func.__qualname__}` must be an async function")

        @wraps(func)
        def wrapper(*args, **kwargs):
            if len(args) < 1 or not isinstance(args[0], Context):
                raise TypeError(
                    "expect a Context as the first positional argument "
                    "(not as a keyword argument)"
                )
            ctx = args[0]

            found = []
            _collect_channels(args, found)
            _collect_channels(kwargs, found)

            async def run() -> None:
                try:
                    await func(*args, **kwargs)
                finally:
                    for ch in (*found, *extra_channels):
                        await ch.shutdown(ctx)

            return PyNode(run())

        return wrapper

    return decorator


def run_streaming_pipeline(*, nodes, py_executor = None):
    """
    Run streaming nodes to completion (blocking).

    Accepts a collection of nodes. Native C++ nodes are moved into the C++ pipeline
    and executed with minimal Python overhead, while Python nodes are gathered and
    executed on a dedicated event loop in the provided executor.

    Parameters
    ----------
    nodes
        Iterable of nodes. Each element is either a native C++ node or a Python
        awaitable representing a node.
    py_executor
        Executor used to run Python nodes (required if any Python nodes are present).
        If no Python nodes are provided, this is ignored.

        ``py_executor`` is set as the default executor on the event loop managed
        by rapidsmpf. If your Python nodes include an
        ``await asyncio.to_thread(blocking_function)`` then the blocking function
        will run in ``py_executor``.

    Warnings
    --------
    C++ nodes are released and must not be used after this call.

    Raises
    ------
    ValueError
        If Python nodes are present but no executor is provided.
    Exception
        Any unhandled exception from a node is re-raised after execution. If multiple
        nodes raise exceptions, only one is re-raised, and it is unspecified which one.
    TypeError
        If nodes contains an unknown node type.

    Examples
    --------
    >>> ch: Channel[TableChunk] = context.create_channel()
    >>> cpp_node, output = pull_from_channel(context, ch_in=ch)
    ...
    >>> @define_py_node()
    ... async def python_node(ctx: Context, ch_out: Channel) -> None:
    ...     # Send one message and close.
    ...     await ch_out.send(
    ...         context,
    ...         Message(TableChunk.from_pylibcudf_table(42, ...))
    ...     )
    ...     await ch_out.drain(context)
    ...
    >>> run_streaming_pipeline(
    ...     nodes=[cpp_node, python_node(context, ch_out=ch)],
    ...     py_executor=ThreadPoolExecutor(max_workers=1),
    ... )
    >>> results = output.release()
    >>> tbl = TableChunk.from_message(results[0])
    >>> tbl.sequence_number
    42
    """

    # Split nodes into C++ nodes and Python nodes.
    cdef vector[cpp_Node] cpp_nodes
    cdef list py_nodes = []
    for node in nodes:
        if isinstance(node, CppNode):
            cpp_nodes.push_back(move(deref((<CppNode>node).release_handle())))
        elif isinstance(node, PyNode):
            py_nodes.append(node)
        else:
            raise ValueError(
                "Unknown node type, did you forget to use `@define_py_node()`?"
            )

    async def runner():
        loop = asyncio.get_running_loop()
        loop.set_default_executor(py_executor)
        return await asyncio.gather(*py_nodes)

    if len(py_nodes) > 0:
        if py_executor is None:
            raise ValueError("must provide a py_executor to run Python nodes.")
        py_future = py_executor.submit(asyncio.run, runner())

    try:
        if cpp_nodes.size() > 0:
            with nogil:
                cpp_run_streaming_pipeline(move(cpp_nodes))
    finally:
        if len(py_nodes) > 0:
            py_future.result()  # This will raise any unhandled exception.
