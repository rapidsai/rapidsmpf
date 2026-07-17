# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio

from rapidsmpf.streaming.core.context cimport Context


async def shutdown_channels(Context ctx not None, *chs):
    """
    Shutdown channels, recording and then propagating any exceptions

    Parameters
    ----------
    ctx
        Streaming context for channel shutdown
    chs
        Channels to shutdown

    Raises
    ------
    Any exceptions that shutting down the channels produces.
    """
    results = await asyncio.gather(
        *(ch.shutdown(ctx) for ch in chs), return_exceptions=True
    )
    errors = [r for r in results if isinstance(r, BaseException)]
    if errors:
        cause = (
            errors[0]
            if len(errors) == 1
            else ExceptionGroup("Errors during shutdown of Channels", errors)
        )
        raise cause


async def await_cpp_future(future, *, on_cancel=None):
    """
    Await a C++ future handling cancellation.

    Parameters
    ----------
    future
        Awaitable future returning None bridging into libcoro.
    on_cancel
        Optional callback to run if cancellation is raised while awaiting
        the future. Must be a zero-argument function that returns an
        awaitable.
    """
    try:
        # This shield makes sure that if a cancellation is raised, the
        # future is not cancelled.
        await asyncio.shield(future)
    except asyncio.CancelledError as cancelled:
        # The outer awaitable was cancelled, but we must still ensure the
        # C++ future still runs to completion.
        errors = []
        if on_cancel is not None:
            try:
                # Run cancellation callback, e.g. to shutdown channels that are live
                await on_cancel()
            except BaseException as error:
                errors.append(error)
        try:
            # This could still fail so we catch and reraise the
            # cancellation but recording the C++ exception as well.
            await asyncio.shield(future)
        except BaseException as error:
            errors.append(error)
        if errors:
            cause = (
                errors[0]
                if len(errors) == 1
                else ExceptionGroup("Errors during cancellation of C++ awaitable", errors)
            )
            raise cancelled from cause
        # Otherwise just reraise the cancellation
        raise
