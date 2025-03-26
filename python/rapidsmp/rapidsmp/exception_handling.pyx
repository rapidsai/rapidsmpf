# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

cdef CppExcept translate_py_to_cpp_exception(py_exception) noexcept:
    """Translate a Python exception into a C++ exception handle (`CppExcept`).

    The returned exception handle can then be thrown by `throw_py_as_cpp_exception()`,
    which MUST be done without holding the GIL.

    This is useful when C++ calls a Python function and needs to catch or
    propagate exceptions.

    Parameters
    ----------
    py_exception
        The Python exception to translate.

    Returns
    -------
    The exception description, which should be given to `throw_py_as_cpp_exception()`
    to throw the C++ exception.
    """

    # We map errors to an index, which must match the order of the C++ switch
    # implementing `throw_py_as_cpp_exception()`.
    errors = [
        MemoryError,
        TypeError,
        ValueError,
        IOError,
        IndexError,
        OverflowError,
        ArithmeticError,
    ]
    for i, error in enumerate(errors):
        if isinstance(py_exception, error):
            return CppExcept(i, str.encode(str(py_exception)))
    # Defaults to `RuntimeError`.
    return CppExcept(-1, str.encode(str(py_exception)))

cdef extern from *:
    """
    #include <ios>
    void cpp_throw_py_as_cpp_exception(std::pair<int, std::string> const &res) {
        switch(res.first) {
            case 0:
                throw std::bad_alloc();
            case 1:
                throw std::bad_cast();
            case 2:
                throw std::invalid_argument(res.second);
            case 3:
                throw std::ios_base::failure(res.second);
            case 4:
                throw std::out_of_range(res.second);
            case 5:
                throw std::overflow_error(res.second);
            case 6:
                throw std::range_error(res.second);
            default:
                throw std::runtime_error(res.second);
        }
    }
    """
    void cpp_throw_py_as_cpp_exception(CppExcept) nogil

cdef void throw_py_as_cpp_exception(CppExcept err) noexcept nogil:
    """Throws the exception specified by `CppExcept` as a C++ exception.

    This function MUST be called without the GIL otherwise the thrown C++ exception
    are translated back into a Python exception.

    Parameters
    ----------
    err
        The exception description that will be thrown as a C++ exception.
    """
    cpp_throw_py_as_cpp_exception(err)
