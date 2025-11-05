# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport shared_ptr
from libcpp.utility cimport move

from rapidsmpf.streaming.core.context cimport Context, cpp_Context
from rapidsmpf.streaming.core.message cimport Message, cpp_Message
from rapidsmpf.streaming.core.utilities cimport cython_invoke_python_function

import asyncio


cdef extern from * nogil:
    """
    namespace {
    coro::task<void> _channel_drain_task(
        std::shared_ptr<rapidsmpf::streaming::Channel> channel,
        std::shared_ptr<rapidsmpf::streaming::Context> ctx,
        void (*py_invoker)(void*),
        void *py_function
    ) {
        co_await channel->drain(ctx->executor());
        py_invoker(py_function);
    }
    }  // namespace

    void cpp_channel_drain(
        std::shared_ptr<rapidsmpf::streaming::Context> ctx,
        std::shared_ptr<rapidsmpf::streaming::Channel> channel,
        void (*py_invoker)(void*),
        void *py_function
    ) {
        RAPIDSMPF_EXPECTS(
            ctx->executor()->spawn(
                _channel_drain_task(
                    std::move(channel), ctx, py_invoker, py_function
                )
            ),
            "could not spawn task on thread pool"
        );
    }
    """
    void cpp_channel_drain(
        shared_ptr[cpp_Context] ctx,
        shared_ptr[cpp_Channel] channel,
        void (*py_invoker)(void*),
        void *py_function
    )


cdef extern from * nogil:
    """
    namespace {
    coro::task<void> _channel_shutdown_task(
        std::shared_ptr<rapidsmpf::streaming::Channel> channel,
        void (*py_invoker)(void*),
        void *py_function
    ) {
        co_await channel->shutdown();
        py_invoker(py_function);
    }
    }  // namespace

    void cpp_channel_shutdown(
        std::shared_ptr<rapidsmpf::streaming::Context> ctx,
        std::shared_ptr<rapidsmpf::streaming::Channel> channel,
        void (*py_invoker)(void*),
        void *py_function
    ) {
        RAPIDSMPF_EXPECTS(
            ctx->executor()->spawn(
                _channel_shutdown_task(
                    std::move(channel), py_invoker, py_function
                )
            ),
            "could not spawn task on thread pool"
        );
    }
    """
    void cpp_channel_shutdown(
        shared_ptr[cpp_Context] ctx,
        shared_ptr[cpp_Channel] channel,
        void (*py_invoker)(void*),
        void *py_function
    )


cdef extern from * nogil:
    """
    namespace {
    coro::task<void> _channel_send_task(
        std::shared_ptr<rapidsmpf::streaming::Channel> channel,
        rapidsmpf::streaming::Message msg,
        void (*py_invoker)(void*),
        void *py_function
    ) {
        co_await channel->send(std::move(msg));
        py_invoker(py_function);
    }
    }  // namespace

    void cpp_channel_send(
        std::shared_ptr<rapidsmpf::streaming::Context> ctx,
        std::shared_ptr<rapidsmpf::streaming::Channel> channel,
        rapidsmpf::streaming::Message msg,
        void (*py_invoker)(void*),
        void *py_function
    ) {
        RAPIDSMPF_EXPECTS(
            ctx->executor()->spawn(
                _channel_send_task(
                    std::move(channel), std::move(msg), py_invoker, py_function
                )
            ),
            "could not spawn task on thread pool"
        );
    }
    """
    void cpp_channel_send(
        shared_ptr[cpp_Context] ctx,
        shared_ptr[cpp_Channel] channel,
        cpp_Message msg,
        void (*py_invoker)(void*),
        void *py_function
    )


cdef extern from * nogil:
    """
    namespace {
    coro::task<void> _channel_recv_task(
        std::shared_ptr<rapidsmpf::streaming::Channel> channel,
        rapidsmpf::streaming::Message &msg_output,
        void (*py_invoker)(void*),
        void *py_function
    ) {
        msg_output = co_await channel->receive();
        py_invoker(py_function);
    }
    }  // namespace

    void cpp_channel_recv(
        std::shared_ptr<rapidsmpf::streaming::Context> ctx,
        std::shared_ptr<rapidsmpf::streaming::Channel> channel,
        rapidsmpf::streaming::Message &msg_output,
        void (*py_invoker)(void*),
        void *py_function
    ) {
        RAPIDSMPF_EXPECTS(
            ctx->executor()->spawn(
                _channel_recv_task(
                    std::move(channel), msg_output, py_invoker, py_function
                )
            ),
            "could not spawn task on thread pool"
        );
    }
    """
    void cpp_channel_recv(
        shared_ptr[cpp_Context] ctx,
        shared_ptr[cpp_Channel] channel,
        cpp_Message &msg_output,
        void (*py_invoker)(void*),
        void *py_function
    )

cdef class Channel:
    """
    A coroutine-based, bounded channel for asynchronously sending and
    receiving `Message` objects.
    """
    def __init__(self):
        raise ValueError(
            "Do not create a channel directly, use `Context.create_channel()`"
        )

    @staticmethod
    cdef from_handle(shared_ptr[cpp_Channel] ch):
        cdef Channel self = Channel.__new__(Channel)
        self._handle = ch
        return self

    def __dealloc__(self):
        with nogil:
            self._handle.reset()

    @classmethod
    def __class_getitem__(cls, args):
        return cls

    async def drain(self, Context ctx not None):
        """
        Drain pending messages and then shut down the channel.

        Parameters
        ----------
        ctx
            The current streaming context.
        """
        loop = asyncio.get_running_loop()
        ret = loop.create_future()

        def set_result():
            loop.call_soon_threadsafe(ret.set_result, None)

        with nogil:
            cpp_channel_drain(
                ctx._handle,
                self._handle,
                cython_invoke_python_function,
                <void *>set_result
            )
        await ret

    async def shutdown(self, Context ctx not None):
        """
        Immediately shut down the channel.

        Completes when the shutdown has been processed.

        Parameters
        ----------
        ctx
            The current streaming context.

        Notes
        -----
        Pending and future ``send``/``recv`` operations will complete with failure.
        """
        loop = asyncio.get_running_loop()
        ret = loop.create_future()

        def set_result():
            loop.call_soon_threadsafe(ret.set_result, None)

        with nogil:
            cpp_channel_shutdown(
                ctx._handle,
                self._handle,
                cython_invoke_python_function,
                <void *>set_result
            )
        await ret

    async def send(self, Context ctx, Message msg not None):
        """
        Send a message into the channel.

        Parameters
        ----------
        ctx
            The current streaming context.
        msg
            Message to move into the channel.

        Warnings
        --------
        `msg` is released and left empty after this call.
        """
        loop = asyncio.get_running_loop()
        ret = loop.create_future()

        def set_result():
            loop.call_soon_threadsafe(ret.set_result, None)

        with nogil:
            cpp_channel_send(
                ctx._handle,
                self._handle,
                move(msg._handle),
                cython_invoke_python_function,
                <void *>set_result
            )
        await ret

    async def recv(self, Context ctx not None):
        """
        Receive the next message from the channel.

        Parameters
        ----------
        ctx
            The current streaming context.

        Returns
        -------
        A `Message` if a message is available, otherwise ``None`` if the channel is
        shut down and empty.
        """
        loop = asyncio.get_running_loop()
        ret = loop.create_future()

        cdef cpp_Message msg_output

        def f():
            if msg_output.empty():
                return ret.set_result(None)

            ret.set_result(
                Message.from_handle(move(msg_output))
            )

        def set_result():
            loop.call_soon_threadsafe(f)

        with nogil:
            cpp_channel_recv(
                ctx._handle,
                self._handle,
                msg_output,
                cython_invoke_python_function,
                <void *>set_result
            )
        return await ret
