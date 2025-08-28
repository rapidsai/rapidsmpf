# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference as deref
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

import asyncio
import inspect
from collections.abc import Awaitable, Iterable, Iterator, Mapping
from functools import wraps

from rapidsmpf.streaming.core.channel import Channel


cdef class CppNode:
    """
    A streaming node (coroutine) implemented in C++.

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

    cdef unique_ptr[cpp_Node] handle_release(self):
        """
        Release and return the underlying native handle.

        Returns
        -------
        The native handle with ownership transferred to the caller.

        Raises
        ------
        ValueError
            If the node is uninitialized or has already been consumed.

        Notes
        -----
        After calling this, the node no longer owns a handle and should not be
        used.
        """
        if not self._handle:
            raise ValueError("CppNode is uninitialized, has it been consumed?")
        return move(self._handle)


class PyNode(Awaitable[None]):
    """
    A streaming node (coroutine) implemented in Python.

    This runs as an Python coroutine (asyncio), which means it comes with a significant
    Python overhead. The GIL is release on `await` and when calling the C++ API.
    """
    def __init__(self, coro: Awaitable[None]) -> None:
        self._coro = coro

    def __await__(self) -> Iterator[None]:
        return self._coro.__await__()


def define_py_node(ctx, *, channels = ()):
    """
    Create a decorator that defines a Python streaming node.

    This factory wraps an async function into a `PyNode`. It ensures that the function
    behaves as a streaming node in the pipeline, and it takes care of shutting down
    channels automatically when the node finishes (whether normally or due to an
    exception).

    Parameters
    ----------
    ctx
        The streaming context of the decorated function.
    channels
        Additional channels to shut down after the decorated function completes. Useful
        when the node uses channels that are not passed as coroutine arguments.

    Returns
    -------
    decorator
        A decorator to apply to an async function, returning a `PyNode`.

    Raises
    ------
    TypeError
        If the decorated function is not declared ``async``.

    Notes
    -----
    - Channels are discovered by recursively inspecting the bound arguments:
      * Direct arguments are inspected.
      * Mapping values and general iterables are traversed.
      * Text-like objects (``str``, ``bytes``, ``bytearray``, ``memoryview``)
        are skipped.

    Examples
    --------
    >>> @define_py_node(ctx, channels=(ch_extra,))
    ... async def my_node(ch_out):
    ...     await ch_out.send(...)
    ...     # both ch_out and ch_extra will be shut down on exit
    """

    def get_channels(arguments):
        ret = []

        def collect(obj) -> None:
            if isinstance(obj, Channel):
                ret.append(obj)
            elif isinstance(obj, (str, bytes, bytearray, memoryview)):
                return
            elif isinstance(obj, Mapping):
                for v in obj.values():
                    collect(v)
            elif isinstance(obj, Iterable):
                for v in obj:
                    collect(v)

        collect(arguments)
        return ret

    def decorator(func):
        if not inspect.iscoroutinefunction(func):
            raise TypeError("can only decorate async functions")

        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Discover channels from the bound arguments at call time.
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            found_channels = get_channels(bound.arguments)

            async def run() -> None:
                try:
                    await func(*args, **kwargs)
                finally:
                    # Always attempt shutdown (found first, then extra)
                    for ch in (*found_channels, *channels):
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

    Warnings
    --------
    C++ nodes are **consumed** (moved) and must not be used after this call.

    Raises
    ------
    ValueError
        If Python nodes are present but no executor is provided.
    Exception
        Any unhandled exception from any node is re-raised after execution. If multiple
        nodes raise unhandled exceptions, only one (unspecified) exception is re-raised.

    Raises
    ------
    TypeError
        If nodes contains an unknown node type.

    Examples
    --------
    >>> ch: Channel[TableChunk] = Channel()
    >>> cpp_node, output = pull_from_channel(ctx=context, ch_in=ch)
    ...
    >>> @define_py_node(context)
    ... async def python_node(ch_out: Channel) -> None:
    ...     # Send one message and close.
    ...     await ch_out.send(
    ...         context,
    ...         Message(TableChunk.from_pylibcudf_table(42, ...))
    ...     )
    ...     await ch_out.drain(context)
    ...
    >>> run_streaming_pipeline(
    ...     nodes=[cpp_node, python_node(ch_out=ch)],
    ...     py_executor=ThreadPoolExecutor(max_workers=1),
    ... )
    >>> results = output.release()
    >>> tbl = TableChunk.from_message(results[0])
    >>> tbl.sequence_number()
    42
    """

    # Split nodes into C++ nodes and Python nodes.
    cdef vector[cpp_Node] cpp_nodes
    cdef list py_nodes = []
    for node in nodes:
        if isinstance(node, CppNode):
            cpp_nodes.push_back(move(deref((<CppNode>node).handle_release())))
        elif isinstance(node, PyNode):
            py_nodes.append(node)
        else:
            raise ValueError(
                "Unknown node type, did you forget to use `@define_py_node()`?"
            )

    async def runner():
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
