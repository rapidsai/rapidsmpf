# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cpython.object cimport PyObject
from cpython.ref cimport Py_INCREF
from cython.operator cimport dereference as deref
from libc.stdint cimport int32_t, uint32_t
from libcpp.memory cimport make_unique, shared_ptr
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
from rapidsmpf.owning_wrapper cimport cpp_OwningWrapper
from rapidsmpf.shuffler cimport (PartitionAssignment,
                                 cpp_insert_chunk_into_partition_map,
                                 cpp_Shuffler)
from rapidsmpf.streaming._detail.libcoro_spawn_task cimport cpp_set_py_future
from rapidsmpf.streaming.chunks.utils cimport py_deleter
from rapidsmpf.streaming.coll.shuffler cimport cpp_ShufflerAsync
from rapidsmpf.streaming.core.actor cimport CppActor, cpp_Actor
from rapidsmpf.streaming.core.channel cimport Channel
from rapidsmpf.streaming.core.context cimport Context, cpp_Context

import asyncio


cdef extern from * nogil:
    """
    namespace {
    coro::task<void> insert_finished_task(
        rapidsmpf::streaming::ShufflerAsync *shuffle
    ) {
        co_await shuffle->insert_finished();
    }

    void cpp_insert_finished(
        std::shared_ptr<rapidsmpf::streaming::Context> ctx,
        rapidsmpf::streaming::ShufflerAsync *shuffle,
        void (*cpp_set_py_future)(void*, const char *),
        rapidsmpf::OwningWrapper py_future
    ) {
        RAPIDSMPF_EXPECTS(
            ctx->executor()->spawn_detached(
                cython_libcoro_task_wrapper(
                    cpp_set_py_future,
                    std::move(py_future),
                    insert_finished_task(shuffle)
                )
            ),
            "libcoro's spawn_detached() failed to spawn task"
        );
    }
    }  // namespace
    """
    void cpp_insert_finished(
        shared_ptr[cpp_Context] ctx,
        cpp_ShufflerAsync *shuffle,
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
    PartitionAssignment partition_assignment = PartitionAssignment.ROUND_ROBIN
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

    Returns
    -------
    A streaming actor that finishes when shuffling is complete and `ch_out` has
    been drained.

    """

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
        )
    return CppActor.from_handle(make_unique[cpp_Actor](move(_ret)), owner=None)


cdef class ShufflerAsync:
    """
    An asynchronous shuffle.

    Parameters
    ----------
    ctx
        Streaming context
    comm
        The communicator the shuffle is collective over.
    op_id
        Operation id identifying this shuffle. Must not be reused while
        this object is still live.
    total_num_partitions
        Global number of output partitions in the shuffle.
    partition_assignment
        How to assign partition IDs to ranks: :attr:`~.PartitionAssignment.ROUND_ROBIN`
        (default) for load balance (e.g. hash shuffle), or
        :attr:`~.PartitionAssignment.CONTIGUOUS` so each rank gets a contiguous range
        of partition IDs (e.g. for sort so concatenation order matches global order).
        A custom callable may be supported in the future.
    """
    def __init__(
        self,
        Context ctx not None,
        Communicator comm not None,
        int32_t op_id,
        uint32_t total_num_partitions,
        PartitionAssignment partition_assignment = PartitionAssignment.ROUND_ROBIN,
    ):
        self._comm = comm
        self._br = ctx.br()
        with nogil:
            self._handle = make_unique[cpp_ShufflerAsync](
                ctx._handle, comm._handle, op_id, total_num_partitions,
                cpp_Shuffler.round_robin
                if partition_assignment == PartitionAssignment.ROUND_ROBIN
                else cpp_Shuffler.contiguous,
            )

    def __dealloc__(self):
        with nogil:
            self._handle.reset()

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
             Map of partition ID to :class:`~PackedData` associated with
             that partition.
        """
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

    async def insert_finished(self, Context ctx not None):
        """
        Insert a finished marker into the shuffle.

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
        cdef span[const uint32_t] _ret
        cdef list partitions = []
        with nogil:
            _ret = deref(self._handle).local_partitions()
        for pid in _ret:
            partitions.append(pid)
        return partitions
