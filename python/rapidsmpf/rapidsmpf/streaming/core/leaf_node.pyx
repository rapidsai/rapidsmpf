# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0


from libcpp.memory cimport make_unique, shared_ptr
from libcpp.utility cimport move

from rapidsmpf.streaming.core.channel cimport (Channel, Message, cpp_Channel,
                                               cpp_Message)
from rapidsmpf.streaming.core.context cimport Context, cpp_Context
from rapidsmpf.streaming.core.node cimport CppNode, cpp_Node


cdef extern from "<rapidsmpf/streaming/core/leaf_node.hpp>" nogil:
    cdef cpp_Node cpp_push_to_channel \
        "rapidsmpf::streaming::node::push_to_channel"(
            shared_ptr[cpp_Context] ctx,
            shared_ptr[cpp_Channel] ch_out,
            vector[cpp_Message] messages,
        ) except +

    cdef cpp_Node cpp_pull_from_channel \
        "rapidsmpf::streaming::node::pull_from_channel"(
            shared_ptr[cpp_Context] ctx,
            shared_ptr[cpp_Channel] ch_in,
            vector[cpp_Message] out_messages,
        ) except +


def push_to_channel(Context ctx, Channel ch_out, object messages):
    """
    Push all messages to an output channel and drain channel.

    Sends each message in order and, when finished, marks the channel as
    complete so downstream receivers observe completion.

    Parameters
    ----------
    ctx
        The current streaming context.
    ch_out
        Output channel that will receive the messages.
    messages
        Iterable of messages to send. Each element is **consumed** (moved)
        into this message.

    Returns
    -------
    Streaming node representing the asynchronous operation.

    Warnings
    --------
    Input messages are consumed and must not be used after this call.

    Raises
    ------
    ValueError
        If any input message is empty.
    """
    cdef vector[cpp_Message] _messages
    owner = []
    for msg in messages:
        owner.append(msg)
        _messages.emplace_back(move((<Message?>msg)._handle))

    cdef cpp_Node _ret
    with nogil:
        _ret = cpp_push_to_channel(
            ctx._handle, ch_out._handle, move(_messages)
        )
    return CppNode.from_handle(make_unique[cpp_Node](move(_ret)), owner)


cdef class DeferredMessages:
    """
    Deferred list for messages populated by `pull_from_channel`.
    """
    def release(self):
        """
        Release and return the collected messages.

        This is not thread-safe.

        Returns
        -------
        A list of all collected messages in send order.
        """
        cdef list ret = []
        for i in range(self._messages.size()):
            ret.append(
                Message.from_handle(handle=move(self._messages[i]))
            )
        self._messages.clear()
        return ret


def pull_from_channel(Context ctx, Channel ch_in):
    """
    Pull all messages from an input channel.

    Receives messages from the channel until it is closed and collects them
    into a deferred container.

    Parameters
    ----------
    ctx
        The current streaming context.
    ch_in
        Input channel providing messages.

    Returns
    -------
    node
        Streaming node representing the asynchronous receive.
    messages
        Deferred collection that will be populated with the received messages
        as the node runs.

    Warnings
    --------
    The returned messages container is initially empty; schedule the node to
    execute in order to populate it.
    """
    cdef DeferredMessages _ret_messages = DeferredMessages()
    cdef cpp_Node _ret_node
    with nogil:
        _ret_node = cpp_pull_from_channel(
            ctx._handle, ch_in._handle, _ret_messages._messages
        )
    return CppNode.from_handle(
        make_unique[cpp_Node](move(_ret_node)), owner=_ret_messages
    ), _ret_messages
