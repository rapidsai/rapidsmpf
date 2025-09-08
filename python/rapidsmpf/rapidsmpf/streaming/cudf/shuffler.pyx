# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint8_t, uint32_t
from libcpp.memory cimport make_unique, shared_ptr
from libcpp.utility cimport move
from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.pylibrmm.stream cimport Stream

from rapidsmpf.streaming.core.channel cimport Channel, cpp_Channel
from rapidsmpf.streaming.core.context cimport Context, cpp_Context
from rapidsmpf.streaming.core.node cimport CppNode, cpp_Node


cdef extern from "<rapidsmpf/streaming/cudf/shuffler.hpp>" nogil:
    cdef cpp_Node cpp_shuffler \
        "rapidsmpf::streaming::node::shuffler"(
            shared_ptr[cpp_Context] ctx,
            cuda_stream_view stream,
            shared_ptr[cpp_Channel] ch_in,
            shared_ptr[cpp_Channel] ch_out,
            uint8_t op_id,
            uint32_t total_num_partitions,
        ) except +


def shuffler(
    Context ctx not None,
    Stream stream not None,
    Channel ch_in not None,
    Channel ch_out not None,
    uint8_t op_id,
    uint32_t total_num_partitions,
):
    """
    Launch a shuffler node for a single shuffle operation.

    Streaming variant of the RapdisMPF shuffler that reads packed, partitioned
    input chunks from an input channel and emits output chunks grouped by
    partition owner.

    Parameters
    ----------
    ctx
        The node context to use.
    stream
        CUDA stream on which to perform the shuffle. If input chunks were
        created on a different stream, appropriate stream synchronization is
        performed.
    ch_in
        Input channel that supplies partitioned map chunks to be shuffled.
    ch_out
        Output channel that receives the grouped (vector) chunks.
    op_id
        Unique identifier for this shuffle operation. Must not be reused until
        all nodes participating in the shuffle have shut down.
    total_num_partitions
        Total number of logical partitions to shuffle the data into.

    Returns
    -------
    A streaming node that finishes when shuffling is complete and `ch_out` has
    been drained.

    Notes
    -----
    Partition ownership is assigned per the underlying C++ implementation's default
    policy (round-robin across ranks/nodes).
    """

    cdef cuda_stream_view _stream = stream.view()
    cdef cpp_Node _ret
    with nogil:
        _ret = cpp_shuffler(
            ctx._handle,
            _stream,
            ch_in._handle,
            ch_out._handle,
            op_id,
            total_num_partitions,
        )
    return CppNode.from_handle(make_unique[cpp_Node](move(_ret)), owner=None)
