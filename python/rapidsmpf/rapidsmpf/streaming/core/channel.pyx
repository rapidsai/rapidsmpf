# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport make_shared, shared_ptr
from libcpp.utility cimport move

from rapidsmpf._detail.exception_handling cimport (
    CppExcept, throw_py_as_cpp_exception, translate_py_to_cpp_exception)
from rapidsmpf.streaming.core.context cimport Context, cpp_Context

import asyncio


cdef void cython_invoke_python_function(void* py_function) noexcept nogil:
    """
    Invokes a Python function from C++ in a Cython-safe manner.

    This function calls a Python function while ensuring proper exception handling.
    If a Python exception occurs, it is translated into a corresponding C++ exception.

    Notice, we use the `noexcept` keyword to make sure Cython doesn't translate the
    C++ function back into a Python function.

    Parameters
    ----------
    py_function
        A Python callable that that takes no arguments and returns None.

    Raises
    ------
    Converts Python exceptions to C++ exceptions using `throw_py_as_cpp_exception`.
    """
    cdef CppExcept err
    with gil:
        try:
            (<object?>py_function)()
            return
        except BaseException as e:
            err = translate_py_to_cpp_exception(e)
    throw_py_as_cpp_exception(err)


cdef class Message:
    """
    A message to be transferred between streaming nodes.

    Parameters
    ----------
    payload
        A payload object that implements the `Payload` protocol. The payload is
        moved into this message.

    Warnings
    --------
    `payload` is released by this call and must not be used afterwards.
    """
    def __init__(self, payload):
        payload.into_message(self)

    @classmethod
    def __class_getitem__(cls, args):
        return cls

    @staticmethod
    cdef from_handle(cpp_Message handle):
        """
        Construct a Message from an existing C++ handle.

        Parameters
        ----------
        handle
            A C++ message handle whose ownership will be **moved** into the
            returned `Message`.

        Returns
        -------
        A new Python `Message` object owning `handle`.
        """
        cdef Message ret = Message.__new__(Message)
        ret._handle = move(handle)
        return ret

    def __dealloc__(self):
        with nogil:
            self._handle.reset()

    def empty(self):
        """
        Return whether this message is empty.

        Returns
        -------
        True if the message is empty; otherwise, False.
        """
        cdef bool_t ret
        with nogil:
            ret = self._handle.empty()
        return ret


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
    def __cinit__(self):
        self._handle = make_shared[cpp_Channel]()

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
