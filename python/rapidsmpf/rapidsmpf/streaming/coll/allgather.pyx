# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cpython.object cimport PyObject
from cpython.ref cimport Py_INCREF
from cython.operator cimport dereference as deref
from libc.stdint cimport int32_t
from libcpp cimport bool
from libcpp.memory cimport make_unique, shared_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector

from rapidsmpf.coll.allgather cimport Ordered as cpp_Ordered
from rapidsmpf.memory.packed_data cimport (PackedData, cpp_PackedData,
                                           packed_data_vector_to_list)
from rapidsmpf.owning_wrapper cimport cpp_OwningWrapper
from rapidsmpf.streaming._detail.libcoro_spawn_task cimport cpp_set_py_future
from rapidsmpf.streaming.chunks.utils cimport py_deleter
from rapidsmpf.streaming.core.channel cimport Channel
from rapidsmpf.streaming.core.context cimport Context, cpp_Context
from rapidsmpf.streaming.core.node cimport CppNode, cpp_Node

import asyncio


cdef extern from * nogil:
    """
    namespace {
    coro::task<void> extract_all_task(
        rapidsmpf::streaming::AllGather *gather,
        rapidsmpf::streaming::AllGather::Ordered ordered,
        std::shared_ptr<std::vector<rapidsmpf::PackedData>> output
    ) {
        *output = co_await gather->extract_all(ordered);
    }

    std::shared_ptr<std::vector<rapidsmpf::PackedData>> cpp_extract_all(
        std::shared_ptr<rapidsmpf::streaming::Context> ctx,
        rapidsmpf::streaming::AllGather *gather,
        rapidsmpf::streaming::AllGather::Ordered ordered,
        void (*cpp_set_py_future)(void*, const char *),
        rapidsmpf::OwningWrapper py_future
    ) {
        auto output = std::make_shared<std::vector<rapidsmpf::PackedData>>();
        RAPIDSMPF_EXPECTS(
            ctx->executor()->spawn_detached(
                cython_libcoro_task_wrapper(
                    cpp_set_py_future,
                    std::move(py_future),
                    extract_all_task(gather, ordered, output)
                )
            ),
            "libcoro's spawn_detached() failed to spawn task"
        );
        return output;
    }
    }  // namespace
    """
    shared_ptr[vector[cpp_PackedData]] cpp_extract_all(
        shared_ptr[cpp_Context] ctx,
        cpp_AllGather *gather,
        cpp_Ordered ordered,
        void (*cpp_set_py_future)(void*, const char *),
        cpp_OwningWrapper py_future
    ) except +ex_handler


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
    def __init__(self, Context ctx not None, int32_t op_id):
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
        ret = asyncio.get_running_loop().create_future()
        Py_INCREF(ret)
        cdef shared_ptr[vector[cpp_PackedData]] c_ret
        with nogil:
            c_ret = cpp_extract_all(
                ctx._handle,
                self._handle.get(),
                cpp_Ordered.YES if ordered else cpp_Ordered.NO,
                cpp_set_py_future,
                move(cpp_OwningWrapper(<void*><PyObject*>ret, py_deleter))
            )
        await ret
        return packed_data_vector_to_list(move(deref(c_ret)))


def allgather(
    Context ctx not None,
    Channel ch_in not None,
    Channel ch_out not None,
    int32_t op_id,
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
