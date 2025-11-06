# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from functools import partial
from cython.operator cimport dereference as deref
from libcpp.memory cimport make_shared, shared_ptr

from rapidsmpf.streaming.core.channel cimport Channel, cpp_Channel
from rapidsmpf.streaming.core.context cimport Context, cpp_Context
from rapidsmpf.streaming.core.utilities cimport cython_invoke_python_function

import asyncio


cdef extern from * nogil:
    """
    namespace {
    coro::task<void> _lineariser_drain_task(
        std::shared_ptr<rapidsmpf::streaming::Lineariser> lineariser,
        void (*py_invoker)(void*),
        void *py_function
    ) {
        co_await lineariser->drain();
        py_invoker(py_function);
    }
    }  // namespace

    void cpp_lineariser_drain(
        std::shared_ptr<rapidsmpf::streaming::Context> ctx,
        std::shared_ptr<rapidsmpf::streaming::Lineariser> lineariser,
        void (*py_invoker)(void*),
        void *py_function
    ) {
        RAPIDSMPF_EXPECTS(
            ctx->executor()->spawn(
                _lineariser_drain_task(
                    std::move(lineariser), py_invoker, py_function
                )
            ),
            "could not spawn task on thread pool"
        );
    }
    """
    void cpp_lineariser_drain(
        shared_ptr[cpp_Context] ctx,
        shared_ptr[cpp_Lineariser] lineariser,
        void (*py_invoker)(void*),
        void *py_function
    )


cdef class Lineariser:
    """
    Linearise a total order on a fixed number of producers into an output channel.

    Parameters
    ----------
    ctx
        Streaming execution context.
    output
        Channel to linearise into.
    num_producers
        Number of producers.
    """
    def __cinit__(
        self, Context ctx not None, Channel output not None, size_t num_producers
    ):
        self._ctx = ctx
        self._handle = make_shared[cpp_Lineariser](
            ctx._handle, output._handle, num_producers
        )

    def __dealloc__(self):
        with nogil:
            self._handle.reset()

    @classmethod
    def __class_getitem__(cls, args):
        return cls

    def get_inputs(self):
        """
        Obtain the input channels

        Returns
        -------
        List of input channels, one per producer.
        """
        cdef vector[shared_ptr[cpp_Channel]]* c_inputs = (
            &deref(self._handle).get_inputs()
        )
        return [
            Channel.from_handle(c_inputs[0][i]) for i in range(c_inputs.size())
        ]

    async def drain(self):
        """
        Drain pending messages and then shut down the lineariser.
        """
        loop = asyncio.get_running_loop()
        ret = loop.create_future()
        callback = partial(loop.call_soon_threadsafe, partial(ret.set_result, None))

        with nogil:
            cpp_lineariser_drain(
                self._ctx._handle,
                self._handle,
                cython_invoke_python_function,
                <void *>callback
            )
        await ret
