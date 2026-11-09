# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cpython.object cimport PyObject
from cpython.ref cimport Py_INCREF
from cython.operator cimport dereference as deref
from libc.stdint cimport int32_t
from libcpp cimport bool
from libcpp.memory cimport make_shared, make_unique, shared_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector

from rapidsmpf.coll.allgather cimport Ordered as cpp_Ordered
from rapidsmpf.communicator.communicator cimport Communicator
from rapidsmpf.memory.packed_data cimport (PackedData, cpp_PackedData,
                                           packed_data_vector_to_list)
from rapidsmpf.owning_wrapper cimport cpp_OwningWrapper
from rapidsmpf.streaming._detail.libcoro_spawn_task cimport cpp_set_py_future
from rapidsmpf.streaming.chunks.utils cimport py_deleter
from rapidsmpf.streaming.core.actor cimport CppActor, cpp_Actor
from rapidsmpf.streaming.core.channel cimport Channel
from rapidsmpf.streaming.core.context cimport Context

import asyncio


cdef extern from * nogil:
    """
    namespace {
    // The task owns a reference, so the allgather outlives the task even if the
    // Python object is freed or the Python awaiter is cancelled.
    coro::task<void> extract_all_task(
        std::shared_ptr<rapidsmpf::streaming::AllGather> gather,
        rapidsmpf::streaming::AllGather::Ordered ordered,
        std::shared_ptr<std::vector<rapidsmpf::PackedData>> output
    ) {
        *output = co_await gather->extract_all(ordered);
    }

    std::shared_ptr<std::vector<rapidsmpf::PackedData>> cpp_extract_all(
        std::shared_ptr<rapidsmpf::streaming::AllGather> gather,
        rapidsmpf::streaming::AllGather::Ordered ordered,
        void (*cpp_set_py_future)(void*, const char *),
        rapidsmpf::OwningWrapper py_future
    ) {
        auto ctx = gather->ctx();
        auto output = std::make_shared<std::vector<rapidsmpf::PackedData>>();
        RAPIDSMPF_EXPECTS(
            ctx->executor()->spawn_detached(
                cython_libcoro_task_wrapper(
                    cpp_set_py_future,
                    std::move(py_future),
                    extract_all_task(std::move(gather), ordered, output)
                )
            ),
            "libcoro's spawn_detached() failed to spawn task"
        );
        return output;
    }

    // Like `extract_all_task()`, the task owns a reference. Spawning `gather->wait()`
    // directly would not keep the allgather alive.
    coro::task<void> wait_task(
        std::shared_ptr<rapidsmpf::streaming::AllGather> gather
    ) {
        co_await gather->wait();
    }

    void cpp_spawn_wait(
        std::shared_ptr<rapidsmpf::streaming::AllGather> gather,
        void (*cpp_set_py_future)(void*, const char *),
        rapidsmpf::OwningWrapper py_future
    ) {
        auto ctx = gather->ctx();
        RAPIDSMPF_EXPECTS(
            ctx->executor()->spawn_detached(
                cython_libcoro_task_wrapper(
                    cpp_set_py_future,
                    std::move(py_future),
                    wait_task(std::move(gather))
                )
            ),
            "libcoro's spawn_detached() failed to spawn task"
        );
    }
    }  // namespace
    """
    shared_ptr[vector[cpp_PackedData]] cpp_extract_all(
        shared_ptr[cpp_AllGather] gather,
        cpp_Ordered ordered,
        void (*cpp_set_py_future)(void*, const char *),
        cpp_OwningWrapper py_future
    ) except +ex_handler
    void cpp_spawn_wait(
        shared_ptr[cpp_AllGather] gather,
        void (*cpp_set_py_future)(void*, const char *),
        cpp_OwningWrapper py_future
    ) except +ex_handler


cdef class AllGather(Collective):
    """
    An asynchronous AllGather.

    Release the allgather with :meth:`aclose`.

    Parameters
    ----------
    ctx
        Streaming context
    comm
        The communicator the allgather is over.
    op_id
        Operation id identifying this allgather. Must not be reused until
        :meth:`aclose` has returned.
    """
    def __init__(self, Context ctx not None, Communicator comm not None, int32_t op_id):
        self._ctx = ctx
        self._comm = comm
        self.in_context = False
        with nogil:
            self._handle = make_shared[cpp_AllGather](
                ctx._handle, comm._handle, op_id
            )

    def __dealloc__(self):
        with nogil:
            self._handle.reset()

    cdef void _cpp_insert_finished(self) except *:
        with nogil:
            deref(self._handle).insert_finished()

    cdef void _cpp_spawn_wait(self, object future) except *:
        Py_INCREF(future)
        with nogil:
            cpp_spawn_wait(
                self._handle,
                cpp_set_py_future,
                move(cpp_OwningWrapper(<void*><PyObject*>future, py_deleter))
            )

    cdef void _cpp_release(self) noexcept:
        with nogil:
            self._handle.reset()

    @property
    def comm(self):
        """
        Get the communicator used by the allgather.

        Returns
        -------
        The communicator.
        """
        return self._comm

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
        self._check_open()
        if not packed_data.c_obj:
            raise ValueError("PackedData was empty")
        with nogil:
            deref(self._handle).insert(sequence_number, move(deref(packed_data.c_obj)))

    def insert_finished(self):
        """
        Insert a finished marker into the AllGather.
        """
        if self.in_context:
            raise ValueError("Cannot call insert_finished() from within a context")
        self._mark_finished()

    def __enter__(self):
        self.in_context = True
        return self

    def __exit__(self, exc_type, exc, tb):
        self.in_context = False
        self.insert_finished()
        return False  # do not suppress exceptions

    async def extract_all(self, Context ctx = None, *, bool ordered):
        """
        Suspend and extract all data from the AllGather.

        Parameters
        ----------
        ctx
            Ignored. The context passed to the constructor is used. This parameter
            will be removed.
        ordered
            Should the extraction be ordered?

        Returns
        -------
        Awaitable that returns the gathered PackedData.
        """
        self._check_open()
        ret = asyncio.get_running_loop().create_future()
        Py_INCREF(ret)
        cdef shared_ptr[vector[cpp_PackedData]] c_ret
        with nogil:
            c_ret = cpp_extract_all(
                self._handle,
                cpp_Ordered.YES if ordered else cpp_Ordered.NO,
                cpp_set_py_future,
                move(cpp_OwningWrapper(<void*><PyObject*>ret, py_deleter))
            )
        await ret
        return packed_data_vector_to_list(move(deref(c_ret)), self._ctx.br())


def allgather(
    Context ctx not None,
    Communicator comm not None,
    Channel ch_in not None,
    Channel ch_out not None,
    int32_t op_id,
    *,
    bool ordered,
):
    """
    Launch an allgather actor for a single allgather operation.

    Streaming variant of the RapidsMPF allgather.

    Parameters
    ----------
    ctx
        The actor context to use.
    comm
        The communicator the allgather is collective over.
    ch_in
        Input channel that supplies PackedDataChunks to be gathered.
    ch_out
        Output channel that receives gathered PackedDataChunks.
    op_id
        Unique (per-communicator) identifier for this allgather operation.
        Must not be reused until all actors participating in the allgather
        have shut down.
    ordered
        Should the output channel provide data in order of input sequence numbers?

    Returns
    -------
    A streaming actor that finishes when the allgather is complete and ``ch_out`` has
    been drained.
    """

    cdef cpp_Actor _ret
    cdef cpp_Ordered c_ordered = cpp_Ordered.YES if ordered else cpp_Ordered.NO
    with nogil:
        _ret = cpp_allgather(
            ctx._handle,
            comm._handle,
            ch_in._handle,
            ch_out._handle,
            op_id,
            c_ordered,
        )
    return CppActor.from_handle(make_unique[cpp_Actor](move(_ret)), owner=None)
