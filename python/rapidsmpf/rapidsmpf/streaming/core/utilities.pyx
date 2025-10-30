# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from rapidsmpf._detail.exception_handling cimport (
    CppExcept, throw_py_as_cpp_exception, translate_py_to_cpp_exception)


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
