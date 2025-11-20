# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint8_t
from libcpp cimport bool
from libcpp.memory cimport make_unique, shared_ptr
from libcpp.utility cimport move

from rapidsmpf.allgather.allgather cimport Ordered as cpp_Ordered
from rapidsmpf.streaming.core.channel cimport Channel, cpp_Channel
from rapidsmpf.streaming.core.context cimport Context, cpp_Context
from rapidsmpf.streaming.core.node cimport CppNode, cpp_Node


cdef extern from "<rapidsmpf/streaming/coll/allgather.hpp>" nogil:
    cdef cpp_Node cpp_allgather \
        "rapidsmpf::streaming::node::allgather"(
            shared_ptr[cpp_Context] ctx,
            shared_ptr[cpp_Channel] ch_in,
            shared_ptr[cpp_Channel] ch_out,
            uint8_t op_id,
            cpp_Ordered ordered,
        ) except +


def allgather(
    Context ctx not None,
    Channel ch_in not None,
    Channel ch_out not None,
    uint8_t op_id,
    *,
    bool ordered,
):
    """
    Launch an allgather node for a single allgather operation.

    Streaming variant of the RapidsMPF allgather.

    Parameters
    ----------
    ctx
        The node context to use.
    ch_in
        Input channel that supplies PackedDataChunks to be gathered.
    ch_out
        Output channel that receives gathered PackedDataChunks.
    op_id
        Unique identifier for this allgather operation. Must not be reused until
        all nodes participating in the allgather have shut down.
    ordered
        Should the output channel provide data in order of input sequence numbers?

    Returns
    -------
    A streaming node that finishes when the allgather is complete and `ch_out` has
    been drained.
    """

    cdef cpp_Node _ret
    cdef cpp_Ordered c_ordered = cpp_Ordered.YES if ordered else cpp_Ordered.NO
    with nogil:
        _ret = cpp_allgather(
            ctx._handle,
            ch_in._handle,
            ch_out._handle,
            op_id,
            c_ordered,
        )
    return CppNode.from_handle(make_unique[cpp_Node](move(_ret)), owner=None)
