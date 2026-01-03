# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cpython.object cimport PyObject
from cpython.ref cimport Py_INCREF
from cython.operator cimport dereference as deref
from libc.stdint cimport uint8_t
from libcpp cimport bool
from libcpp.memory cimport make_unique, shared_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector

import asyncio
from functools import partial

from rapidsmpf.coll.allgather cimport Ordered as cpp_Ordered
from rapidsmpf.memory.packed_data cimport (PackedData, cpp_PackedData,
                                           packed_data_vector_to_list)
from rapidsmpf.owning_wrapper cimport cpp_OwningWrapper
from rapidsmpf.streaming.chunks.utils cimport py_deleter
from rapidsmpf.streaming.core.channel cimport Channel
from rapidsmpf.streaming.core.context cimport Context, cpp_Context
from rapidsmpf.streaming.core.node cimport CppNode, cpp_Node
from rapidsmpf.streaming.core.utilities cimport cython_invoke_python_function


cdef extern from * nogil:
    """
    namespace {
    coro::task<void> _extract_all_task(
        rapidsmpf::streaming::AllGather *gather,
        rapidsmpf::streaming::AllGather::Ordered ordered,
        std::vector<rapidsmpf::PackedData> &output,
        void (*py_invoker)(void*),
        rapidsmpf::OwningWrapper py_callback
    ) {
        output = co_await gather->extract_all(ordered);
        py_invoker(py_callback.get());
    }

    void cpp_extract_all(
        std::shared_ptr<rapidsmpf::streaming::Context> ctx,
        rapidsmpf::streaming::AllGather *gather,
        rapidsmpf::streaming::AllGather::Ordered ordered,
        std::vector<rapidsmpf::PackedData> &output,
        void (*py_invoker)(void*),
        rapidsmpf::OwningWrapper py_callback
    ) {
        RAPIDSMPF_EXPECTS(
            ctx->executor()->spawn_detached(
                 _extract_all_task(
                     gather, ordered, output, py_invoker, std::move(py_callback)
                 )
            ),
            "could not spawn task on thread pool"
        );
    }
    }
    """
    void cpp_extract_all(
        shared_ptr[cpp_Context] ctx,
        cpp_AllGather *gather,
        cpp_Ordered ordered,
        vector[cpp_PackedData] &output,
        void (*py_invoker)(void*),
        cpp_OwningWrapper py_callback,
    ) except +


cdef class AllGather:
    """
    An asynchronous AllGather.

    Parameters
    ----------
    ctx
        Streaming context
    op_id
        Operation id identifying this allgather. Must not be reused while
        this object is still live.
    """
    def __init__(self, Context ctx not None, uint8_t op_id):
        with nogil:
            self._handle = make_unique[cpp_AllGather](
                ctx._handle, op_id
            )

    def __dealloc__(self):
        with nogil:
            self._handle.reset()

    def insert(self, uint64_t sequence_number, PackedData packed_data not None):
        """
        Insert data into the AllGather.

        Parameters
        ----------
        sequence_number
            Sequence number of this piece of data, used to provide an
            ordering when extracting.
        packed_data
            The data to insert.
        """
        if not packed_data.c_obj:
            raise ValueError("PackedData was empty")
        with nogil:
            deref(self._handle).insert(sequence_number, move(deref(packed_data.c_obj)))

    def insert_finished(self):
        """
        Insert a finished marker into the AllGather.
        """
        with nogil:
            deref(self._handle).insert_finished()

    async def extract_all(self, Context ctx, *, bool ordered):
        """
        Suspend and extract all data from the AllGather.

        Parameters
        ----------
        ctx
            Streaming context.
        ordered
            Should the extraction be ordered?

        Returns
        -------
        Awaitable that returns the gathered PackedData.
        """
        loop = asyncio.get_running_loop()
        ret = loop.create_future()
        callback = partial(loop.call_soon_threadsafe, partial(ret.set_result, None))
        cdef vector[cpp_PackedData] c_ret
        Py_INCREF(callback)
        with nogil:
            cpp_extract_all(
                ctx._handle,
                self._handle.get(),
                cpp_Ordered.YES if ordered else cpp_Ordered.NO,
                c_ret,
                cython_invoke_python_function,
                move(cpp_OwningWrapper(<void*><PyObject*>callback, py_deleter))
            )
        await ret
        return packed_data_vector_to_list(move(c_ret))


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
