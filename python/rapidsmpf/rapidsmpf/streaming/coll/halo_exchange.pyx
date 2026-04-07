# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cpython.object cimport PyObject
from cpython.ref cimport Py_INCREF
from cython.operator cimport dereference as deref
from libc.stdint cimport int32_t
from libcpp.memory cimport make_unique, shared_ptr, unique_ptr
from libcpp.optional cimport make_optional, nullopt, optional
from libcpp.pair cimport pair
from libcpp.utility cimport move

from rapidsmpf.communicator.communicator cimport Communicator
from rapidsmpf.memory.packed_data cimport PackedData, cpp_PackedData
from rapidsmpf.owning_wrapper cimport cpp_OwningWrapper
from rapidsmpf.streaming._detail.libcoro_spawn_task cimport cpp_set_py_future
from rapidsmpf.streaming.chunks.utils cimport py_deleter
from rapidsmpf.streaming.core.context cimport Context, cpp_Context

import asyncio


# Output type: pair of nullable unique_ptr (nullptr == absent direction).
# Using unique_ptr avoids optional<PackedData> on the Cython side, which
# would require moving through a FakeReference (a Cython limitation).
cdef extern from * nogil:
    """
    namespace {

    // Convert optional<PackedData> → unique_ptr<PackedData> (nullptr if absent).
    static std::unique_ptr<rapidsmpf::PackedData> opt_to_uptr(
        std::optional<rapidsmpf::PackedData> opt
    ) {
        return opt.has_value()
            ? std::make_unique<rapidsmpf::PackedData>(std::move(*opt))
            : nullptr;
    }

    // Coroutine task that drives one exchange() round and stores the result.
    coro::task<void> exchange_task(
        rapidsmpf::streaming::HaloExchange* he,
        std::optional<rapidsmpf::PackedData> send_left,
        std::optional<rapidsmpf::PackedData> send_right,
        std::shared_ptr<std::pair<
            std::unique_ptr<rapidsmpf::PackedData>,
            std::unique_ptr<rapidsmpf::PackedData>
        >> output
    ) {
        auto result =
            co_await he->exchange(std::move(send_left), std::move(send_right));
        output->first  = opt_to_uptr(std::move(result.first));
        output->second = opt_to_uptr(std::move(result.second));
    }

    // Spawn the exchange task on the executor; when done, set the Python future.
    std::shared_ptr<std::pair<
        std::unique_ptr<rapidsmpf::PackedData>,
        std::unique_ptr<rapidsmpf::PackedData>
    >>
    cpp_exchange(
        std::shared_ptr<rapidsmpf::streaming::Context> ctx,
        rapidsmpf::streaming::HaloExchange* he,
        std::optional<rapidsmpf::PackedData> send_left,
        std::optional<rapidsmpf::PackedData> send_right,
        void (*cpp_set_py_future)(void*, const char*),
        rapidsmpf::OwningWrapper py_future
    ) {
        auto output = std::make_shared<std::pair<
            std::unique_ptr<rapidsmpf::PackedData>,
            std::unique_ptr<rapidsmpf::PackedData>
        >>();
        RAPIDSMPF_EXPECTS(
            ctx->executor()->spawn_detached(
                cython_libcoro_task_wrapper(
                    cpp_set_py_future,
                    std::move(py_future),
                    exchange_task(he,
                                  std::move(send_left),
                                  std::move(send_right),
                                  output)
                )
            ),
            "libcoro's spawn_detached() failed to spawn task"
        );
        return output;
    }

    }  // namespace
    """
    shared_ptr[pair[unique_ptr[cpp_PackedData], unique_ptr[cpp_PackedData]]] cpp_exchange(  # noqa: E501
        shared_ptr[cpp_Context] ctx,
        cpp_HaloExchange* he,
        optional[cpp_PackedData] send_left,
        optional[cpp_PackedData] send_right,
        void (*cpp_set_py_future)(void*, const char*),
        cpp_OwningWrapper py_future,
    ) except +ex_handler


cdef class HaloExchange:
    """
    Point-to-point halo exchange between adjacent ranks.

    Each call to :meth:`exchange` performs one bidirectional neighbor round:
    rank k sends to rank k+1 (received as ``from_left`` by k+1) and to
    rank k-1 (received as ``from_right`` by k-1).

    Parameters
    ----------
    ctx : Context
        Streaming context.
    comm : Communicator
        Communicator.
    op_id : int
        Pre-allocated operation ID (uses stages 0–3).
    """

    def __init__(self, Context ctx not None, Communicator comm not None, int32_t op_id):
        self._ctx = ctx
        self._comm = comm
        with nogil:
            self._handle = make_unique[cpp_HaloExchange](
                ctx._handle, comm._handle, op_id
            )

    def __dealloc__(self):
        with nogil:
            self._handle.reset()

    async def exchange(
        self,
        send_left:  PackedData | None,
        send_right: PackedData | None,
    ):
        """
        Perform one round of bidirectional neighbor exchange.

        Parameters
        ----------
        send_left : PackedData or None
            Data to send to rank-1; ``None`` for boundary or no data.
        send_right : PackedData or None
            Data to send to rank+1; ``None`` for boundary or no data.

        Returns
        -------
        tuple[PackedData | None, PackedData | None]
            ``(from_left, from_right)`` — data received from rank-1 and
            rank+1 respectively; ``None`` if absent or neighbor sent nothing.

        Notes
        -----
        Calls to ``exchange()`` on a single instance must be sequential:
        the next call may only be issued after the previous ``await``
        completes.  Concurrent calls corrupt per-round state.

        The ``HaloExchange`` instance must not be destroyed while an
        ``exchange()`` call is pending.
        """
        cdef optional[cpp_PackedData] c_send_right
        cdef optional[cpp_PackedData] c_send_left

        if send_left is not None:
            if not send_left.c_obj:
                raise ValueError("send_left PackedData was empty")
            c_send_left = make_optional[cpp_PackedData](move(deref(send_left.c_obj)))
        else:
            c_send_left = nullopt

        if send_right is not None:
            if not send_right.c_obj:
                raise ValueError("send_right PackedData was empty")
            c_send_right = make_optional[cpp_PackedData](move(deref(send_right.c_obj)))
        else:
            c_send_right = nullopt

        ret = asyncio.get_running_loop().create_future()
        Py_INCREF(ret)
        cdef shared_ptr[pair[unique_ptr[cpp_PackedData], unique_ptr[cpp_PackedData]]] c_ret  # noqa: E501
        with nogil:
            c_ret = cpp_exchange(
                self._ctx._handle,
                self._handle.get(),
                move(c_send_left),
                move(c_send_right),
                cpp_set_py_future,
                move(cpp_OwningWrapper(<void*><PyObject*>ret, py_deleter))
            )
        await ret

        # Extract unique_ptr results; nullptr == absent direction.
        cdef unique_ptr[cpp_PackedData] ul = move(deref(c_ret).first)
        cdef unique_ptr[cpp_PackedData] ur = move(deref(c_ret).second)
        from_left = PackedData.from_librapidsmpf(move(ul)) if ul else None
        from_right = PackedData.from_librapidsmpf(move(ur)) if ur else None
        return from_left, from_right
