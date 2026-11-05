# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cpython.object cimport PyObject
from cpython.ref cimport Py_INCREF
from cython.operator cimport dereference as deref
from libc.stdint cimport int32_t
from libcpp.memory cimport make_shared, shared_ptr
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
from rapidsmpf.streaming.core.context cimport Context


cdef extern from * nogil:
    """
    namespace {
    // The task owns a reference, so the exchange outlives the task even if the
    // Python object is freed or the Python awaiter is cancelled.
    coro::task<void> wait_task(
        std::shared_ptr<rapidsmpf::streaming::SparseAlltoall> exchange
    ) {
        co_await exchange->wait();
    }

    void cpp_spawn_wait(
        std::shared_ptr<rapidsmpf::streaming::SparseAlltoall> exchange,
        void (*cpp_set_py_future)(void*, const char *),
        rapidsmpf::OwningWrapper py_future
    ) {
        auto ctx = exchange->ctx();
        RAPIDSMPF_EXPECTS(
            ctx->executor()->spawn_detached(
                cython_libcoro_task_wrapper(
                    cpp_set_py_future,
                    std::move(py_future),
                    wait_task(std::move(exchange))
                )
            ),
            "libcoro's spawn_detached() failed to spawn task"
        );
    }
    }  // namespace
    """
    void cpp_spawn_wait(
        shared_ptr[cpp_SparseAlltoall] exchange,
        void (*cpp_set_py_future)(void*, const char *),
        cpp_OwningWrapper py_future
    ) except +ex_handler


cdef class SparseAlltoall(Collective):
    """
    An asynchronous sparse all-to-all.

    Release the collective with :meth:`aclose`, or use it as an async context
    manager, which calls :meth:`aclose` on exit.

    Parameters
    ----------
    ctx
        Streaming context.
    comm
        The communicator the collective is over.
    op_id
        Operation id identifying this sparse all-to-all. Must not be reused until
        :meth:`aclose` has returned.
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
        self._ctx = ctx
        self._comm = comm
        self._br = ctx.br()
        cdef vector[Rank] c_srcs = list(srcs)
        cdef vector[Rank] c_dsts = list(dsts)
        with nogil:
            self._handle = make_shared[cpp_SparseAlltoall](
                ctx._handle,
                comm._handle,
                op_id,
                move(c_srcs),
                move(c_dsts),
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

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        # When the body raised, a missing insert_finished() is a consequence of that
        # error, so only the body's exception propagates.
        await self._aclose(exc_type is None)
        return False

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
        self._check_open()
        if not packed_data.c_obj:
            raise ValueError("PackedData was empty")
        with nogil:
            deref(self._handle).insert(dst, move(deref(packed_data.c_obj)))

    async def insert_finished(self, Context ctx = None):
        """
        Insert the finish markers and wait for the collective to complete on this rank.

        Errors from inserting the finish markers are raised before the first
        suspension. If the awaiting task is cancelled, the collective keeps running
        and :meth:`aclose` waits for it.

        Parameters
        ----------
        ctx
            Ignored. The context passed to the constructor is used. This parameter
            will be removed.

        Notes
        -----
        This must be awaited before extraction can occur.
        """
        self._mark_finished()
        await self._wait()

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
        self._check_open()
        cdef vector[cpp_PackedData] c_ret
        with nogil:
            c_ret = deref(self._handle).extract(src)
        return packed_data_vector_to_list(move(c_ret), self._br)
