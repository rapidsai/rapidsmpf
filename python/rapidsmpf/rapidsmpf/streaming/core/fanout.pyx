# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport make_unique
from libcpp.utility cimport move
from libcpp.vector cimport vector

from rapidsmpf.streaming.core.channel cimport Channel
from rapidsmpf.streaming.core.context cimport Context
from rapidsmpf.streaming.core.fanout cimport FanoutPolicy
from rapidsmpf.streaming.core.node cimport CppNode, cpp_Node


def fanout(Context ctx, Channel ch_in, chs_out, FanoutPolicy policy):
    """
    Broadcast messages from one input channel to multiple output channels.

    The node continuously receives messages from the input channel and forwards
    them to all output channels according to the selected fanout policy.

    Each output channel receives a shallow copy of the same message; no payload
    data is duplicated. All copies share the same underlying payload, ensuring
    zero-copy broadcast semantics.

    Parameters
    ----------
    ctx
        The node context to use.
    ch_in
        Input channel from which messages are received.
    chs_out
        Output channels to which messages are broadcast.
    policy
        The fanout policy to use. `FanoutPolicy.BOUNDED` can be used if all
        output channels are being consumed by independent consumers in the
        downstream. `FanoutPolicy.UNBOUNDED` can be used if the output channels
        are being consumed by a single/ shared consumer in the downstream.
    Returns
    -------
    Streaming node representing the fanout operation.

    Raises
    ------
    ValueError
        If an unknown fanout policy is specified.

    Notes
    -----
    Since messages are shallow-copied, releasing a payload (``release<T>()``)
    is only valid on messages that hold exclusive ownership of the payload.

    >>> import rapidsmpf.streaming.core as streaming
    >>> with streaming.Context(...) as ctx:
    ...     ch_in = ctx.create_channel()
    ...     ch_out1 = ctx.create_channel()
    ...     ch_out2 = ctx.create_channel()
    ...     node = streaming.fanout(
    ...         ctx,
    ...         ch_in,
    ...         [ch_out1, ch_out2],
    ...         streaming.FanoutPolicy.BOUNDED,
    ...     )
    """
    cdef vector[shared_ptr[cpp_Channel]] _chs_out
    if len(chs_out) == 0:
        raise ValueError("output channels cannot be empty")
    owner = []
    for ch_out in chs_out:
        if not isinstance(ch_out, Channel):
            raise TypeError("All elements in chs_out must be Channel instances")
        owner.append(ch_out)
        _chs_out.push_back((<Channel>ch_out)._handle)

    cdef cpp_Node _ret
    with nogil:
        _ret = cpp_fanout(
            ctx._handle, ch_in._handle, move(_chs_out), policy
        )
    return CppNode.from_handle(make_unique[cpp_Node](move(_ret)), owner)
