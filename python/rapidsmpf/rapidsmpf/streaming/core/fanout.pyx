# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum

from libcpp.memory cimport make_unique, shared_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector

from rapidsmpf.streaming.core.channel cimport Channel, cpp_Channel
from rapidsmpf.streaming.core.context cimport Context, cpp_Context
from rapidsmpf.streaming.core.node cimport CppNode, cpp_Node


class FanoutPolicy(IntEnum):
    """
    Fanout policy controlling how messages are propagated.

    Attributes
    ----------
    BOUNDED : int
        Process messages as they arrive and immediately forward them.
        Messages are forwarded as soon as they are received from the input channel.
        The next message is not processed until all output channels have completed
        sending the current one, ensuring backpressure and synchronized flow.
    UNBOUNDED : int
        Forward messages without enforcing backpressure.
        In this mode, messages may be accumulated internally before being
        broadcast, or they may be forwarded immediately depending on the
        implementation and downstream consumption rate.

        This mode disables coordinated backpressure between outputs, allowing
        consumers to process at independent rates, but can lead to unbounded
        buffering and increased memory usage.

        Note: Consumers might not receive any messages until *all* upstream
        messages have been sent, depending on the implementation and buffering
        strategy.
    """
    BOUNDED = <uint8_t>cpp_FanoutPolicy.BOUNDED
    UNBOUNDED = <uint8_t>cpp_FanoutPolicy.UNBOUNDED


def fanout(Context ctx, Channel ch_in, list chs_out, policy):
    """
    Broadcast messages from one input channel to multiple output channels.

    The node continuously receives messages from the input channel and forwards
    them to all output channels according to the selected fanout policy.

    Each output channel receives a shallow copy of the same message; no payload
    data is duplicated. All copies share the same underlying payload, ensuring
    zero-copy broadcast semantics.

    Parameters
    ----------
    ctx : Context
        The node context to use.
    ch_in : Channel
        Input channel from which messages are received.
    chs_out : list[Channel]
        Output channels to which messages are broadcast.
    policy : FanoutPolicy
        The fanout strategy to use (see FanoutPolicy).

    Returns
    -------
    CppNode
        Streaming node representing the fanout operation.

    Raises
    ------
    ValueError
        If an unknown fanout policy is specified.

    Notes
    -----
    Since messages are shallow-copied, releasing a payload (``release<T>()``)
    is only valid on messages that hold exclusive ownership of the payload.

    Examples
    --------
    >>> import rapidsmpf.streaming.core as streaming
    >>> ctx = streaming.Context(...)
    >>> ch_in = ctx.create_channel()
    >>> ch_out1 = ctx.create_channel()
    >>> ch_out2 = ctx.create_channel()
    >>> node = streaming.fanout(
    ...     ctx, ch_in, [ch_out1, ch_out2], streaming.FanoutPolicy.BOUNDED
    ... )
    """
    # Validate policy
    if not isinstance(policy, (FanoutPolicy, int)):
        raise TypeError(f"policy must be a FanoutPolicy enum value, got {type(policy)}")
    
    cdef vector[shared_ptr[cpp_Channel]] _chs_out
    owner = []
    for ch_out in chs_out:
        if not isinstance(ch_out, Channel):
            raise TypeError("All elements in chs_out must be Channel instances")
        owner.append(ch_out)
        _chs_out.push_back((<Channel>ch_out)._handle)

    cdef cpp_FanoutPolicy _policy = <cpp_FanoutPolicy>(<int>policy)
    cdef cpp_Node _ret
    with nogil:
        _ret = cpp_fanout(
            ctx._handle, ch_in._handle, move(_chs_out), _policy
        )
    return CppNode.from_handle(make_unique[cpp_Node](move(_ret)), owner)

