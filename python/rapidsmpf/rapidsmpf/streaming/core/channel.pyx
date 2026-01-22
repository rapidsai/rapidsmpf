# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cpython.object cimport PyObject
from cpython.ref cimport Py_INCREF
from cython.operator cimport dereference as deref
from libcpp.memory cimport shared_ptr
from libcpp.utility cimport move

from rapidsmpf.owning_wrapper cimport cpp_OwningWrapper
from rapidsmpf.streaming._detail.libcoro_spawn_task cimport cpp_set_py_future
from rapidsmpf.streaming.chunks.utils cimport py_deleter
from rapidsmpf.streaming.core.context cimport Context, cpp_Context
from rapidsmpf.streaming.core.message cimport Message, cpp_Message

import asyncio


cdef extern from * nogil:
    """
    namespace {
    coro::task<void> _channel_drain_task(
        std::shared_ptr<rapidsmpf::streaming::Channel> channel,
        std::shared_ptr<rapidsmpf::streaming::Context> ctx
    ) {
        co_await channel->drain(ctx->executor());
    }
    }  // namespace

    void cpp_channel_drain(
        std::shared_ptr<rapidsmpf::streaming::Context> ctx,
        std::shared_ptr<rapidsmpf::streaming::Channel> channel,
        void (*cpp_set_py_future)(void*, const char *),
        rapidsmpf::OwningWrapper py_future
    ) {
        RAPIDSMPF_EXPECTS(
            ctx->executor()->spawn_detached(
                cython_libcoro_task_wrapper(
                    cpp_set_py_future,
                    std::move(py_future),
                    _channel_drain_task(std::move(channel), ctx)
                )
            ),
            "could not spawn task on thread pool"
        );
    }
    """
    void cpp_channel_drain(
        shared_ptr[cpp_Context] ctx,
        shared_ptr[cpp_Channel] channel,
        void (*cpp_set_py_future)(void*, const char *),
        cpp_OwningWrapper py_future
    )


cdef extern from * nogil:
    """
    namespace {
    coro::task<void> _channel_shutdown_task(
        std::shared_ptr<rapidsmpf::streaming::Channel> channel
    ) {
        co_await channel->shutdown();
    }
    }  // namespace

    void cpp_channel_shutdown(
        std::shared_ptr<rapidsmpf::streaming::Context> ctx,
        std::shared_ptr<rapidsmpf::streaming::Channel> channel,
        void (*cpp_set_py_future)(void*, const char *),
        rapidsmpf::OwningWrapper py_future
    ) {
        RAPIDSMPF_EXPECTS(
            ctx->executor()->spawn_detached(
                cython_libcoro_task_wrapper(
                    cpp_set_py_future,
                    std::move(py_future),
                    _channel_shutdown_task(std::move(channel))
                )
            ),
            "could not spawn task on thread pool"
        );
    }
    """
    void cpp_channel_shutdown(
        shared_ptr[cpp_Context] ctx,
        shared_ptr[cpp_Channel] channel,
        void (*cpp_set_py_future)(void*, const char *),
        cpp_OwningWrapper py_future
    )


cdef extern from * nogil:
    """
    namespace {
    coro::task<void> _channel_send_task(
        std::shared_ptr<rapidsmpf::streaming::Channel> channel,
        rapidsmpf::streaming::Message msg
    ) {
        co_await channel->send(std::move(msg));
    }
    }  // namespace

    void cpp_channel_send(
        std::shared_ptr<rapidsmpf::streaming::Context> ctx,
        std::shared_ptr<rapidsmpf::streaming::Channel> channel,
        rapidsmpf::streaming::Message msg,
        void (*cpp_set_py_future)(void*, const char *),
        rapidsmpf::OwningWrapper py_future
    ) {
        RAPIDSMPF_EXPECTS(
            ctx->executor()->spawn_detached(
                cython_libcoro_task_wrapper(
                    cpp_set_py_future,
                    std::move(py_future),
                    _channel_send_task(
                        std::move(channel),
                        std::move(msg)
                    )
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
        void (*cpp_set_py_future)(void*, const char *),
        cpp_OwningWrapper py_future
    )


cdef extern from * nogil:
    """
    namespace {
    coro::task<void> _channel_recv_task(
        std::shared_ptr<rapidsmpf::streaming::Channel> channel,
        std::shared_ptr<rapidsmpf::streaming::Message> msg_output
    ) {
        *msg_output = co_await channel->receive();
    }
    }  // namespace

    std::shared_ptr<rapidsmpf::streaming::Message> cpp_channel_recv(
        std::shared_ptr<rapidsmpf::streaming::Context> ctx,
        std::shared_ptr<rapidsmpf::streaming::Channel> channel,
        void (*cpp_set_py_future)(void*, const char *),
        rapidsmpf::OwningWrapper py_future
    ) {
        auto msg_output = std::make_shared<rapidsmpf::streaming::Message>();
        RAPIDSMPF_EXPECTS(
            ctx->executor()->spawn_detached(
                cython_libcoro_task_wrapper(
                    cpp_set_py_future,
                    std::move(py_future),
                    _channel_recv_task(
                        std::move(channel),
                        msg_output
                    )
                )
            ),
            "could not spawn task on thread pool"
        );
        return msg_output;
    }
    """
    shared_ptr[cpp_Message] cpp_channel_recv(
        shared_ptr[cpp_Context] ctx,
        shared_ptr[cpp_Channel] channel,
        void (*cpp_set_py_future)(void*, const char *),
        cpp_OwningWrapper py_future
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
        ret = asyncio.get_running_loop().create_future()
        Py_INCREF(ret)
        with nogil:
            cpp_channel_drain(
                ctx._handle,
                self._handle,
                cpp_set_py_future,
                move(cpp_OwningWrapper(<void*><PyObject*>ret, py_deleter)),
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
        ret = asyncio.get_running_loop().create_future()
        Py_INCREF(ret)
        with nogil:
            cpp_channel_shutdown(
                ctx._handle,
                self._handle,
                cpp_set_py_future,
                move(cpp_OwningWrapper(<void*><PyObject*>ret, py_deleter)),
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
        ret = asyncio.get_running_loop().create_future()
        Py_INCREF(ret)
        with nogil:
            cpp_channel_send(
                ctx._handle,
                self._handle,
                move(msg._handle),
                cpp_set_py_future,
                move(cpp_OwningWrapper(<void*><PyObject*>ret, py_deleter)),
            )
        await ret

    async def recv(self, Context ctx not None):
        """
        CI-testing
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
        ret = asyncio.get_running_loop().create_future()
        Py_INCREF(ret)
        cdef shared_ptr[cpp_Message] c_msg
        with nogil:
            c_msg = cpp_channel_recv(
                ctx._handle,
                self._handle,
                cpp_set_py_future,
                move(cpp_OwningWrapper(<void*><PyObject*>ret, py_deleter))
            )
        await ret
        if deref(c_msg).empty():
            return None
        return Message.from_handle(move(deref(c_msg)))
