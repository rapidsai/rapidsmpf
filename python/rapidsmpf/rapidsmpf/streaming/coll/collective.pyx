# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Abstract base class of the streaming collectives."""

import asyncio


cdef class Collective:
    """
    Abstract base class of the streaming collectives.

    :class:`~rapidsmpf.streaming.coll.ShufflerAsync`,
    :class:`~rapidsmpf.streaming.coll.SparseAlltoall` and
    :class:`~rapidsmpf.streaming.coll.AllGather` derive from this class, which
    provides :meth:`aclose`. It cannot be instantiated directly.
    """
    def __cinit__(self, *args, **kwargs):
        if type(self) is Collective:
            raise TypeError(
                "Collective is an abstract base class and cannot be instantiated"
            )

    cdef void _check_open(self) except *:
        if self._closed:
            raise RuntimeError(f"{type(self).__name__} is closed")

    cdef void _mark_finished(self) except *:
        self._check_open()
        self._cpp_insert_finished()
        self._finishing = True

    cdef void _cpp_insert_finished(self) except *:
        raise NotImplementedError

    cdef void _cpp_spawn_wait(self, object future) except *:
        raise NotImplementedError

    cdef void _cpp_release(self) noexcept:
        pass

    async def _wait(self):
        if self._complete is not None and self._complete.done():
            # The future may belong to an event loop that has since been closed.
            self._complete.result()
            return
        loop = asyncio.get_running_loop()
        if self._complete is None or self._complete.get_loop() is not loop:
            # A pending future of another event loop cannot be awaited here, so
            # start a new wait on the running loop.
            future = loop.create_future()
            self._cpp_spawn_wait(future)
            self._complete = future
        # A cancelled waiter must not cancel the shared future.
        await asyncio.shield(self._complete)

    async def _aclose(self, bint raise_if_skipped):
        if self._closed:
            return
        if not self._finishing:
            self._mark_finished()
            self._skipped_insert_finished = True
        await self._wait()
        self._cpp_release()
        self._closed = True
        if raise_if_skipped and self._skipped_insert_finished:
            raise RuntimeError("aclose() called before insert_finished()")

    async def aclose(self):
        """
        Wait for the collective to complete on this rank and release it.

        Data that was not extracted is freed. After ``aclose()`` has returned on a
        rank, the ``op_id`` may be reused on that rank. Calling ``aclose()`` again
        has no effect, and every other method raises ``RuntimeError``.

        If ``insert_finished()`` was not called, ``aclose()`` inserts the finish
        markers so that the other ranks can complete, releases the collective and
        then raises ``RuntimeError``.

        ``aclose()`` waits for every rank to insert its finish markers. If another
        rank never does, ``aclose()`` waits forever. Cancelling ``aclose()`` returns
        control while the collective keeps running in the background. Awaiting
        ``aclose()`` again waits for the same completion. Until an ``aclose()`` call
        has returned, the ``op_id`` must not be reused, and after a cancelled
        ``aclose()`` whose collective never completes, the communicator must not be
        used for new collectives.

        Raises
        ------
        RuntimeError
            If ``insert_finished()`` was not called before ``aclose()``.
        ValueError
            If called on an :class:`~rapidsmpf.streaming.coll.AllGather` inside its
            ``with`` block.
        """
        await self._aclose(True)
