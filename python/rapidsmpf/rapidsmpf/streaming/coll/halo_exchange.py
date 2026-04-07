# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Python prototype of HaloExchange for point-to-point halo exchange.

Uses two :class:`~rapidsmpf.streaming.coll.allgather.AllGather`
collectives as a stand-in for direct P2P messaging (O(N) messages).
The target C++ class in ``halo_exchange.hpp`` replaces this with
direct ``Communicator::send/recv`` (O(1) messages per rank).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from rapidsmpf.memory.packed_data import PackedData
from rapidsmpf.streaming.coll.allgather import AllGather

if TYPE_CHECKING:
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.context import Context


class HaloExchange:
    """
    Point-to-point halo exchange between adjacent ranks.

    Exchanges boundary data between neighboring ranks in a linear rank
    topology.  Each call to :meth:`exchange` performs one round of
    bidirectional neighbor exchange:

    - Rank k sends ``send_right`` to rank k+1, which receives it as
      ``from_left``.
    - Rank k sends ``send_left`` to rank k-1, which receives it as
      ``from_right``.

    Multiple calls to :meth:`exchange` on the same instance are safe
    and represent successive rounds of halo propagation (for multi-hop
    window coverage when the rolling-window period spans more than one
    rank's data range).

    .. note::
        This is a Python prototype that uses two
        :class:`~rapidsmpf.streaming.coll.allgather.AllGather`
        collectives as a transport.  The target C++ implementation
        (``halo_exchange.hpp``) replaces those with direct
        ``Communicator::send/recv`` calls, reducing the per-rank message
        count from O(N) to O(1).

    Parameters
    ----------
    ctx : Context
        Streaming context.
    comm : Communicator
        Communicator.
    op_id : int
        Pre-allocated operation ID.  Uses ``op_id`` for the rightward
        AllGather and ``op_id + 1`` for the leftward AllGather.
        Callers must reserve **2 consecutive** op IDs per
        ``HaloExchange`` instance.
    """

    def __init__(self, ctx: Context, comm: Communicator, op_id: int) -> None:
        self._ctx = ctx
        self._comm = comm
        self._op_id = op_id

    async def exchange(
        self,
        send_right: PackedData | None,
        send_left: PackedData | None,
    ) -> tuple[PackedData | None, PackedData | None]:
        """
        Perform one round of bidirectional neighbor exchange.

        Parameters
        ----------
        send_right : PackedData or None
            Data to send to rank+1.  Pass ``None`` when this rank has
            no right neighbor (``rank == nranks - 1``) or has nothing
            to send rightward.
        send_left : PackedData or None
            Data to send to rank-1.  Pass ``None`` when this rank has
            no left neighbor (``rank == 0``) or has nothing to send
            leftward.

        Returns
        -------
        from_left : PackedData or None
            Data received from rank-1; ``None`` if ``rank == 0`` or the
            left neighbor sent nothing rightward.
        from_right : PackedData or None
            Data received from rank+1; ``None`` if ``rank == nranks-1``
            or the right neighbor sent nothing leftward.

        Notes
        -----
        Successive calls are safe because each round awaits completion
        of both AllGather operations before returning, so op_id reuse
        across rounds cannot cause message interleaving.

        A ``PackedData`` whose GPU-data portion is empty (zero bytes) is
        treated as a sentinel meaning "nothing to send".  Callers should
        pass ``None`` rather than an empty ``PackedData`` to signal an
        absent halo.
        """
        ctx = self._ctx
        rank = self._comm.rank
        nranks = self._comm.nranks
        br = ctx.br()

        # Rightward AllGather: every rank inserts its send_right (or a fresh sentinel)
        # with seq_num=rank so extract_all(ordered=True) gives result[k] = rank k.
        # A sentinel is a host-only PackedData with 0 data bytes; detected via
        # `to_host_bytes() == b""` on the receive side.
        # Each insert() moves the PackedData into the AllGather, so sentinels must
        # be freshly allocated — reusing the same object across two inserts is UB.
        right_gather = AllGather(ctx, self._comm, self._op_id)
        right_gather.insert(
            rank,
            send_right if send_right is not None else PackedData.from_host_bytes(b"", br),
        )
        right_gather.insert_finished()

        # Leftward AllGather: same pattern with op_id+1.
        left_gather = AllGather(ctx, self._comm, self._op_id + 1)
        left_gather.insert(
            rank,
            send_left if send_left is not None else PackedData.from_host_bytes(b"", br),
        )
        left_gather.insert_finished()

        # Drive both AllGathers to completion in parallel.
        right_results, left_results = await asyncio.gather(
            right_gather.extract_all(ctx, ordered=True),
            left_gather.extract_all(ctx, ordered=True),
        )

        # from_left = what rank-1 sent rightward (right_results[rank-1])
        from_left: PackedData | None = None
        if rank > 0:
            payload = right_results[rank - 1]
            if payload.to_host_bytes() != b"":
                from_left = payload

        # from_right = what rank+1 sent leftward (left_results[rank+1])
        from_right: PackedData | None = None
        if rank < nranks - 1:
            payload = left_results[rank + 1]
            if payload.to_host_bytes() != b"":
                from_right = payload

        return from_left, from_right
