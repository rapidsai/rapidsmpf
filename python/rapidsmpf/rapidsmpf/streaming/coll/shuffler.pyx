# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from cpython.object cimport PyObject
from cpython.ref cimport Py_INCREF
from cython.operator cimport dereference as deref
from libc.stdint cimport int32_t, uint32_t
from libcpp.memory cimport make_shared, make_unique, shared_ptr
from libcpp.span cimport span
from libcpp.unordered_map cimport unordered_map
from libcpp.utility cimport move
from libcpp.vector cimport vector

from rapidsmpf._detail.exception_handling cimport ex_handler
from rapidsmpf.communicator.communicator cimport Communicator
# Need the header include for inline C++ code
from rapidsmpf.memory.buffer_resource cimport BufferResource  # no-cython-lint
from rapidsmpf.memory.packed_data cimport (PackedData, cpp_PackedData,
                                           packed_data_vector_to_list)
from rapidsmpf.memory.reservation_events cimport (ReservationContext,
                                                  cpp_ReservationContext,
                                                  unwrap_reservation_context)
from rapidsmpf.owning_wrapper cimport cpp_OwningWrapper
from rapidsmpf.shuffler cimport (PartitionAssignment,
                                 cpp_insert_chunk_into_partition_map,
                                 cpp_Shuffler)
from rapidsmpf.streaming._detail.libcoro_spawn_task cimport cpp_set_py_future
from rapidsmpf.streaming.chunks.utils cimport py_deleter
from rapidsmpf.streaming.coll.shuffler cimport cpp_ShufflerAsync
from rapidsmpf.streaming.core.actor cimport CppActor, cpp_Actor
from rapidsmpf.streaming.core.channel cimport Channel
from rapidsmpf.streaming.core.context cimport Context


cdef extern from * nogil:
    """
    namespace {
    // The task owns a reference, so the shuffler outlives the task even if the
    // Python object is freed or the Python awaiter is cancelled.
    coro::task<void> wait_task(
        std::shared_ptr<rapidsmpf::streaming::ShufflerAsync> shuffle
    ) {
        co_await shuffle->wait();
    }

    void cpp_spawn_wait(
        std::shared_ptr<rapidsmpf::streaming::ShufflerAsync> shuffle,
        void (*cpp_set_py_future)(void*, const char *),
        rapidsmpf::OwningWrapper py_future
    ) {
        auto ctx = shuffle->ctx();
        RAPIDSMPF_EXPECTS(
            ctx->executor()->spawn_detached(
                cython_libcoro_task_wrapper(
                    cpp_set_py_future,
                    std::move(py_future),
                    wait_task(std::move(shuffle))
                )
            ),
            "libcoro's spawn_detached() failed to spawn task"
        );
    }
    }  // namespace
    """
    void cpp_spawn_wait(
        shared_ptr[cpp_ShufflerAsync] shuffle,
        void (*cpp_set_py_future)(void*, const char *),
        cpp_OwningWrapper py_future
    ) except +ex_handler


def shuffler(
    Context ctx not None,
    Communicator comm not None,
    Channel ch_in not None,
    Channel ch_out not None,
    int32_t op_id,
    uint32_t total_num_partitions,
    PartitionAssignment partition_assignment = PartitionAssignment.ROUND_ROBIN,
    *,
    ReservationContext reservation_context=None,
):
    """
    Launch a shuffler actor for a single shuffle operation.

    Streaming variant of the RapidsMPF shuffler that reads packed, partitioned
    input chunks from an input channel and emits output chunks grouped by
    partition owner.

    Parameters
    ----------
    ctx
        The actor context to use.
    comm
        The communicator the shuffle is collective over.
    ch_in
        Input channel that supplies partitioned map chunks to be shuffled.
    ch_out
        Output channel that receives the grouped (vector) chunks.
    op_id
        Unique identifier for this shuffle operation. Must not be reused until
        all actors participating in the shuffle have shut down.
    total_num_partitions
        Total number of logical partitions to shuffle the data into.
    partition_assignment
        How to assign partition IDs to ranks: :attr:`~.PartitionAssignment.ROUND_ROBIN`
        (default) for load balance (e.g. hash shuffle), or
        :attr:`~.PartitionAssignment.CONTIGUOUS` so each rank gets a contiguous range
        of partition IDs (e.g. for sort so concatenation order matches global order).
        A custom callable may be supported in the future.
    reservation_context
        Optional telemetry attribution token for internal memory reservations.

    Returns
    -------
    A streaming actor that finishes when shuffling is complete and ``ch_out`` has
    been drained.
    """

    cdef cpp_ReservationContext c_reservation_context = (
        unwrap_reservation_context(reservation_context)
    )
    with nogil:
        _ret = cpp_shuffler(
            ctx._handle,
            comm._handle,
            ch_in._handle,
            ch_out._handle,
            op_id,
            total_num_partitions,
            cpp_Shuffler.round_robin
            if partition_assignment == PartitionAssignment.ROUND_ROBIN
            else cpp_Shuffler.contiguous,
            c_reservation_context,
        )
    return CppActor.from_handle(make_unique[cpp_Actor](move(_ret)), owner=None)


cdef class ShufflerAsync(Collective):
    """
    An asynchronous shuffle.

    Release the shuffle with :meth:`aclose`, or use it as an async context manager,
    which calls :meth:`aclose` on exit.

    Parameters
    ----------
    ctx
        Streaming context
    comm
        The communicator the shuffle is collective over.
    op_id
        Operation id identifying this shuffle. Must not be reused until
        :meth:`aclose` has returned.
    total_num_partitions
        Global number of output partitions in the shuffle.
    partition_assignment
        How to assign partition IDs to ranks: :attr:`~.PartitionAssignment.ROUND_ROBIN`
        (default) for load balance (e.g. hash shuffle), or
        :attr:`~.PartitionAssignment.CONTIGUOUS` so each rank gets a contiguous range
        of partition IDs (e.g. for sort so concatenation order matches global order).
        A custom callable may be supported in the future.
    reservation_context
        Optional telemetry attribution token for internal memory reservations.
    """
    def __init__(
        self,
        Context ctx not None,
        Communicator comm not None,
        int32_t op_id,
        uint32_t total_num_partitions,
        PartitionAssignment partition_assignment = PartitionAssignment.ROUND_ROBIN,
        *,
        ReservationContext reservation_context=None,
    ):
        self._ctx = ctx
        self._comm = comm
        self._br = ctx.br()
        cdef cpp_ReservationContext c_reservation_context = (
            unwrap_reservation_context(reservation_context)
        )
        with nogil:
            self._handle = make_shared[cpp_ShufflerAsync](
                ctx._handle, comm._handle, op_id, total_num_partitions,
                cpp_Shuffler.round_robin
                if partition_assignment == PartitionAssignment.ROUND_ROBIN
                else cpp_Shuffler.contiguous,
                c_reservation_context,
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
        Get the communicator used by the shuffler.

        Returns
        -------
        The communicator.
        """
        return self._comm

    def insert(self, chunks):
        """
        Insert data into the shuffle.

        Parameters
        ----------
        chunks
             Map of partition ID to :class:`~.PackedData` associated with
             that partition.
        """
        self._check_open()
        cdef unordered_map[uint32_t, cpp_PackedData] c_chunks
        c_chunks.reserve(len(chunks))
        for pid, chunk in chunks.items():
            if not (<PackedData?>chunk).c_obj:
                raise ValueError("PackedData was empty")
            cpp_insert_chunk_into_partition_map(
                c_chunks, pid, move((<PackedData>chunk).c_obj)
            )
        with nogil:
            deref(self._handle).insert(move(c_chunks))

    async def insert_finished(self, Context ctx = None):
        """
        Insert the finish markers and wait for the shuffle to complete on this rank.

        Errors from inserting the finish markers are raised before the first
        suspension. If the awaiting task is cancelled, the shuffle keeps running and
        :meth:`aclose` waits for it.

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

    def extract(self, uint32_t pid):
        """
        Extract all chunks belonging to the specified partition.

        Must only be called after awaiting :meth:`insert_finished`.

        Parameters
        ----------
        pid
            The partition to extract.

        Returns
        -------
        list[PackedData]
            The PackedData chunks associated with the partition.
        """
        self._check_open()
        cdef vector[cpp_PackedData] c_ret
        with nogil:
            c_ret = deref(self._handle).extract(pid)
        return packed_data_vector_to_list(move(c_ret), self._br)

    def local_partitions(self):
        """
        Return the partition IDs owned by this rank.

        Returns
        -------
        Partition IDs owned by this shuffler.
        """
        self._check_open()
        cdef span[const uint32_t] _ret
        cdef list partitions = []
        with nogil:
            _ret = deref(self._handle).local_partitions()
        for pid in _ret:
            partitions.append(pid)
        return partitions
