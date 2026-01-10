# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import asyncio


async def _set_result(future):
    future.set_result(None)


async def _set_exception(future, exc):
    future.set_exception(exc)


cdef void cpp_set_py_future(
    void* py_future, const char *error_msg
) noexcept nogil:
    """
    Set the result or exception on an asyncio Future from C++ code.

    This function is intended to be called from C++. It safely schedules
    completion of a Python asyncio Future on its associated event loop
    using thread-safe coroutine submission.

    The function is used together with ``cython_libcoro_task_wrapper``,
    which awaits a C++ ``coro::task`` and invokes this callback upon
    completion or failure. On successful completion, the Future is resolved
    with ``None``. If an exception is thrown by the C++ task, the Future is
    completed with a ``RuntimeError`` containing the exception message.

    Parameters
    ----------
    py_future
        Opaque pointer to a Python ``asyncio.Future`` object.
    error_msg
        Optional C string describing an error. If ``NULL``, the Future
        is resolved successfully. Otherwise, the Future is completed
        with an exception constructed from this message.
    """
    with gil:
        future = (<object?> py_future)
        if error_msg == NULL:
            asyncio.run_coroutine_threadsafe(_set_result(future), future.get_loop())
        else:
            asyncio.run_coroutine_threadsafe(
                _set_exception(future, RuntimeError(error_msg.decode("utf-8"))),
                future.get_loop()
            )
