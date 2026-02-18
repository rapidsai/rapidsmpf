# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cpython.object cimport PyObject
from cpython.ref cimport Py_INCREF
from cython.operator cimport dereference as deref
from libc.stdint cimport int32_t, uint32_t
from libcpp.memory cimport make_unique, shared_ptr
from libcpp.optional cimport optional
from libcpp.span cimport span
from libcpp.unordered_map cimport unordered_map
from libcpp.utility cimport move, pair
from libcpp.vector cimport vector

from rapidsmpf._detail.exception_handling cimport ex_handler
from rapidsmpf.memory.packed_data cimport (PackedData, cpp_PackedData,
                                           packed_data_vector_to_list)
from rapidsmpf.owning_wrapper cimport cpp_OwningWrapper
from rapidsmpf.shuffler cimport cpp_insert_chunk_into_partition_map
from rapidsmpf.streaming._detail.libcoro_spawn_task cimport cpp_set_py_future
from rapidsmpf.streaming.chunks.utils cimport py_deleter
from rapidsmpf.streaming.core.actor cimport CppActor, cpp_Actor
from rapidsmpf.streaming.core.channel cimport Channel
from rapidsmpf.streaming.core.context cimport Context, cpp_Context

import asyncio


cdef extern from * nogil:
    """
    namespace {
    coro::task<void> extract_async_task(
        rapidsmpf::streaming::ShufflerAsync *shuffle,
        std::uint32_t pid,
        std::shared_ptr<std::optional<std::vector<rapidsmpf::PackedData>>> output
    ) {
        *output = co_await shuffle->extract_async(pid);
    }

    std::shared_ptr<std::optional<std::vector<rapidsmpf::PackedData>>>
    cpp_extract_async(
        std::shared_ptr<rapidsmpf::streaming::Context> ctx,
        rapidsmpf::streaming::ShufflerAsync *shuffle,
        std::uint32_t pid,
        void (*cpp_set_py_future)(void*, const char *),
        rapidsmpf::OwningWrapper py_future
    ) {
        auto output = std::make_shared<
            std::optional<std::vector<rapidsmpf::PackedData>>
        >();
        RAPIDSMPF_EXPECTS(
            ctx->executor()->spawn_detached(
                cython_libcoro_task_wrapper(
                    cpp_set_py_future,
                    std::move(py_future),
                    extract_async_task(shuffle, pid, output)
                )
            ),
            "libcoro's spawn_detached() failed to spawn task"
        );
        return output;
    }

    coro::task<void> extract_any_async_task(
        rapidsmpf::streaming::ShufflerAsync *shuffle,
        std::shared_ptr<
            std::optional<std::pair<std::uint32_t, std::vector<rapidsmpf::PackedData>>>
        > output
    ) {
        *output = co_await shuffle->extract_any_async();
    }

    std::shared_ptr<
        std::optional<std::pair<std::uint32_t, std::vector<rapidsmpf::PackedData>>>
    > cpp_extract_any_async(
        std::shared_ptr<rapidsmpf::streaming::Context> ctx,
        rapidsmpf::streaming::ShufflerAsync *shuffle,
        void (*cpp_set_py_future)(void*, const char *),
        rapidsmpf::OwningWrapper py_future
    ) {
        auto output = std::make_shared<
            std::optional<std::pair<std::uint32_t, std::vector<rapidsmpf::PackedData>>>
        >();
        RAPIDSMPF_EXPECTS(
            ctx->executor()->spawn_detached(
                cython_libcoro_task_wrapper(
                    cpp_set_py_future,
                    std::move(py_future),
                    extract_any_async_task(
                        shuffle, output
                    )
                )
            ),
            "libcoro's spawn_detached() failed to spawn task"
        );
        return output;
    }

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
    shared_ptr[optional[vector[cpp_PackedData]]] cpp_extract_async(
        shared_ptr[cpp_Context] ctx,
        cpp_ShufflerAsync *shuffle,
        uint32_t pid,
        void (*cpp_set_py_future)(void*, const char *),
        cpp_OwningWrapper py_future
    ) except +ex_handler

    shared_ptr[optional[pair[uint32_t, vector[cpp_PackedData]]]] \
        cpp_extract_any_async(
        shared_ptr[cpp_Context] ctx,
        cpp_ShufflerAsync *shuffle,
        void (*cpp_set_py_future)(void*, const char *),
        cpp_OwningWrapper py_future
    ) except +ex_handler

    void cpp_insert_finished(
        shared_ptr[cpp_Context] ctx,
        cpp_ShufflerAsync *shuffle,
        void (*cpp_set_py_future)(void*, const char *),
        cpp_OwningWrapper py_future
    ) except +ex_handler


def shuffler(
    Context ctx not None,
    Channel ch_in not None,
    Channel ch_out not None,
    int32_t op_id,
    uint32_t total_num_partitions,
):
    """
    Launch a shuffler actor for a single shuffle operation.

    Streaming variant of the RapdisMPF shuffler that reads packed, partitioned
    input chunks from an input channel and emits output chunks grouped by
    partition owner.

    Parameters
    ----------
    ctx
        The actor context to use.
    ch_in
        Input channel that supplies partitioned map chunks to be shuffled.
    ch_out
        Output channel that receives the grouped (vector) chunks.
    op_id
        Unique identifier for this shuffle operation. Must not be reused until
        all actors participating in the shuffle have shut down.
    total_num_partitions
        Total number of logical partitions to shuffle the data into.

    Returns
    -------
    A streaming actor that finishes when shuffling is complete and `ch_out` has
    been drained.

    Notes
    -----
    Partition ownership is assigned per the underlying C++ implementation's default
    policy (round-robin across ranks/nodes).
    """

    cdef cpp_Actor _ret
    with nogil:
        _ret = cpp_shuffler(
            ctx._handle,
            ch_in._handle,
            ch_out._handle,
            op_id,
            total_num_partitions,
        )
    return CppActor.from_handle(make_unique[cpp_Actor](move(_ret)), owner=None)


cdef class ShufflerAsync:
    """
    An asynchronous shuffle.

    Parameters
    ----------
    ctx
        Streaming context
    op_id
        Operation id identifying this shuffle. Must not be reused while
        this object is still live.
    total_num_partitions
        Global number of output partitions in the shuffle.
    """
    def __init__(
        self, Context ctx not None, int32_t op_id, uint32_t total_num_partitions
    ):
        with nogil:
            self._handle = make_unique[cpp_ShufflerAsync](
                ctx._handle, op_id, total_num_partitions
            )

    def __dealloc__(self):
        with nogil:
            self._handle.reset()

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

    async def extract_async(self, Context ctx not None, uint32_t pid):
        """
        Suspend and extract a partition from the shuffle.

        Parameters
        ----------
        ctx
            Streaming context.
        pid
            The partition to extract.

        Returns
        -------
        list[PackedData]
            The PackedData representing the extracted partition.
        None
            If the partition has already been extracted.
        """
        ret = asyncio.get_running_loop().create_future()
        Py_INCREF(ret)
        cdef shared_ptr[optional[vector[cpp_PackedData]]] c_ret
        with nogil:
            c_ret = cpp_extract_async(
                ctx._handle,
                self._handle.get(),
                pid,
                cpp_set_py_future,
                move(cpp_OwningWrapper(<void*><PyObject*>ret, py_deleter))
            )
        await ret
        if deref(c_ret).has_value():
            return packed_data_vector_to_list(move(deref(deref(c_ret))))
        else:
            return None

    async def extract_any_async(self, Context ctx not None):
        """
        Suspend and extract any partition from the shuffle.

        Parameters
        ----------
        ctx
            Streaming context.

        Returns
        -------
        tuple[int, list[PackedData]]
            The identifier for the extracted partition and the PackedData
            of the partition.
        None
            If there are no more partitions to extract.
        """
        ret = asyncio.get_running_loop().create_future()
        Py_INCREF(ret)
        cdef shared_ptr[optional[pair[uint32_t, vector[cpp_PackedData]]]] c_ret
        with nogil:
            c_ret = cpp_extract_any_async(
                ctx._handle,
                self._handle.get(),
                cpp_set_py_future,
                move(cpp_OwningWrapper(<void*><PyObject*>ret, py_deleter))
            )
        await ret
        if deref(c_ret).has_value():
            return (
                deref(c_ret).value().first,
                packed_data_vector_to_list(move(deref(c_ret).value().second))
            )
        else:
            return None

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
