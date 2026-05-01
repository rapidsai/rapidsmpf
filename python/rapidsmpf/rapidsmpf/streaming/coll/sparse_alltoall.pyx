# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cpython.object cimport PyObject
from cpython.ref cimport Py_INCREF
from cython.operator cimport dereference as deref
from libc.stdint cimport int32_t
from libcpp.memory cimport make_unique, shared_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector

from rapidsmpf._detail.exception_handling cimport ex_handler
from rapidsmpf.communicator.communicator cimport Communicator, Rank
from rapidsmpf.memory.packed_data cimport (PackedData, cpp_PackedData,
                                           packed_data_vector_to_list)
from rapidsmpf.owning_wrapper cimport cpp_OwningWrapper
from rapidsmpf.streaming._detail.libcoro_spawn_task cimport cpp_set_py_future
from rapidsmpf.streaming.chunks.utils cimport py_deleter
from rapidsmpf.streaming.coll.sparse_alltoall cimport cpp_SparseAlltoall
from rapidsmpf.streaming.core.context cimport Context, cpp_Context

import asyncio


cdef extern from * nogil:
    """
    namespace {
    coro::task<void> insert_finished_task(
        rapidsmpf::streaming::SparseAlltoall *exchange
    ) {
        co_await exchange->insert_finished();
    }

    void cpp_insert_finished(
        std::shared_ptr<rapidsmpf::streaming::Context> ctx,
        rapidsmpf::streaming::SparseAlltoall *exchange,
        void (*cpp_set_py_future)(void*, const char *),
        rapidsmpf::OwningWrapper py_future
    ) {
        RAPIDSMPF_EXPECTS(
            ctx->executor()->spawn_detached(
                cython_libcoro_task_wrapper(
                    cpp_set_py_future,
                    std::move(py_future),
                    insert_finished_task(exchange)
                )
            ),
            "libcoro's spawn_detached() failed to spawn task"
        );
    }
    }  // namespace
    """
    void cpp_insert_finished(
        shared_ptr[cpp_Context] ctx,
        cpp_SparseAlltoall *exchange,
        void (*cpp_set_py_future)(void*, const char *),
        cpp_OwningWrapper py_future
    ) except +ex_handler


cdef class SparseAlltoall:
    """
    An asynchronous sparse all-to-all.

    Parameters
    ----------
    ctx
        Streaming context.
    comm
        The communicator the collective is over.
    op_id
        Operation id identifying this sparse all-to-all. Must not be reused while
        this object is still live.
    srcs
        Source ranks this rank receives from.
    dsts
        Destination ranks this rank sends to.
    """
    def __init__(
        self,
        Context ctx not None,
        Communicator comm not None,
        int32_t op_id,
        srcs,
        dsts,
    ):
        self._comm = comm
        self._br = ctx.br()
        cdef vector[Rank] c_srcs = list(srcs)
        cdef vector[Rank] c_dsts = list(dsts)
        with nogil:
            self._handle = make_unique[cpp_SparseAlltoall](
                ctx._handle,
                comm._handle,
                op_id,
                move(c_srcs),
                move(c_dsts),
            )

    def __dealloc__(self):
        with nogil:
            self._handle.reset()

    @property
    def comm(self):
        """
        Get the communicator used by the sparse all-to-all.

        Returns
        -------
        The communicator.
        """
        return self._comm

    def insert(self, Rank dst, PackedData packed_data not None):
        """
        Insert data into the sparse all-to-all.

        Parameters
        ----------
        dst
            Destination rank to send to.
        packed_data
            The data to insert.
        """
        if not packed_data.c_obj:
            raise ValueError("PackedData was empty")
        with nogil:
            deref(self._handle).insert(dst, move(deref(packed_data.c_obj)))

    async def insert_finished(self, Context ctx not None):
        """
        Insert the finished marker and await local completion.

        Notes
        -----
        This must be awaited before extraction can occur.
        """
        ret = asyncio.get_running_loop().create_future()
        Py_INCREF(ret)
        with nogil:
            cpp_insert_finished(
                ctx._handle,
                self._handle.get(),
                cpp_set_py_future,
                move(cpp_OwningWrapper(<void*><PyObject*>ret, py_deleter))
            )
        await ret

    def extract(self, Rank src):
        """
        Extract all data received from the specified source rank.

        Must only be called after awaiting :meth:`insert_finished`.

        Parameters
        ----------
        src
            Source rank to extract from.

        Returns
        -------
        list[PackedData]
            The PackedData messages received from the source.
        """
        cdef vector[cpp_PackedData] c_ret
        with nogil:
            c_ret = deref(self._handle).extract(src)
        return packed_data_vector_to_list(move(c_ret), self._br)
