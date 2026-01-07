# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from rapidsmpf.owning_wrapper cimport cpp_OwningWrapper


cdef extern from * nogil:
    """
    #include <iostream>
    #include <coro/task.hpp>

    /**
     * @brief Await a C++ coro::task and notify a Python asyncio.Future on completion.
     *
     * This is a C++-only helper used by Cython bindings to bridge C++ coroutines
     * and Python asyncio. The function awaits the provided C++ coro::task and,
     * upon completion, invokes the supplied callback to resolve a Python
     * asyncio.Future.
     *
     * The callback is expected to be a Cython-exposed function such as
     * cpp_set_py_future, which safely schedules completion of the Future on
     * its associated event loop.
     *
     * On successful completion of the C++ task, the callback is invoked with
     * a null error message. If the task throws a std::exception, the exception
     * message is forwarded to the callback.
     *
     * @param cpp_set_py_future Callback used to resolve or fail the Python
     * asyncio.Future.
     * @param py_future Owning wrapper holding the Python asyncio.Future.
     * @param task C++ coroutine task whose completion is being bridged to Python.
     *
     * @return A coro::task that completes after the underlying task has finished
     * and the Python Future has been notified.
     */
    coro::task<void> cython_libcoro_task_wrapper(
        void (*cpp_set_py_future)(void*, const char *),
        rapidsmpf::OwningWrapper py_future,
        coro::task<void> task
    ) {
        try{
            co_await task;
            cpp_set_py_future(py_future.get(), NULL);
        } catch(std::exception const& e) {
            cpp_set_py_future(py_future.get(), e.what());
        }
    }
    """

cdef void cpp_set_py_future(void* py_future, const char *error_msg) noexcept nogil
