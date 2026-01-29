# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cpython.ref cimport PyObject
from rapidsmpf.error import BadAlloc, OutOfMemory, ReservationError


# Store references to exception classes for use in C++ code
cdef PyObject* _ReservationError = <PyObject*>ReservationError
cdef PyObject* _OutOfMemory = <PyObject*>OutOfMemory
cdef PyObject* _BadAlloc = <PyObject*>BadAlloc

cdef extern from *:
    """
    enum class ExceptionType {
        RuntimeError = -1,
        MemoryError = 0,
        TypeError = 1,
        ValueError = 2,
        IOError = 3,
        IndexError = 4,
        OverflowError = 5,
        ArithmeticError = 6,
        ReservationError = 7,
        OutOfMemory = 8,
        BadAlloc = 9
    };
    """
    cdef enum class ExceptionType:
        RuntimeError
        MemoryError
        TypeError
        ValueError
        IOError
        IndexError
        OverflowError
        ArithmeticError
        ReservationError
        OutOfMemory
        BadAlloc

# Define mapping between Python exceptions and ExceptionType values
# NOTE: Order matters! More specific exceptions must come before base classes.
cdef dict exception_map = {
    ReservationError: ExceptionType.ReservationError,
    OutOfMemory: ExceptionType.OutOfMemory,
    BadAlloc: ExceptionType.BadAlloc,
    MemoryError: ExceptionType.MemoryError,
    TypeError: ExceptionType.TypeError,
    ValueError: ExceptionType.ValueError,
    IOError: ExceptionType.IOError,
    IndexError: ExceptionType.IndexError,
    OverflowError: ExceptionType.OverflowError,
    ArithmeticError: ExceptionType.ArithmeticError,
}

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
    for py_exc, exc_type in exception_map.items():
        if isinstance(py_exception, py_exc):
            return CppExcept(<int>exc_type, str.encode(str(py_exception)))
    # Defaults to `RuntimeError`.
    return CppExcept(<int>ExceptionType.RuntimeError, str.encode(str(py_exception)))

cdef extern from *:
    """
    #include <ios>
    #include <rapidsmpf/error.hpp>
    void cpp_throw_py_as_cpp_exception(std::pair<int, std::string> const &res) {
        switch(res.first) {
            case static_cast<int>(ExceptionType::ReservationError):
                throw rapidsmpf::reservation_error(res.second);
            case static_cast<int>(ExceptionType::OutOfMemory):
                throw rapidsmpf::out_of_memory(res.second);
            case static_cast<int>(ExceptionType::BadAlloc):
                throw rapidsmpf::bad_alloc(res.second);
            case static_cast<int>(ExceptionType::MemoryError):
                throw std::bad_alloc();
            case static_cast<int>(ExceptionType::TypeError):
                throw std::bad_cast();
            case static_cast<int>(ExceptionType::ValueError):
                throw std::invalid_argument(res.second);
            case static_cast<int>(ExceptionType::IOError):
                throw std::ios_base::failure(res.second);
            case static_cast<int>(ExceptionType::IndexError):
                throw std::out_of_range(res.second);
            case static_cast<int>(ExceptionType::OverflowError):
                throw std::overflow_error(res.second);
            case static_cast<int>(ExceptionType::ArithmeticError):
                throw std::range_error(res.second);
            default:  // ExceptionType::RuntimeError
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


# Exception handler for mapping C++ exceptions to Python exceptions
cdef extern from *:
    """
    #include <exception>
    #include <stdexcept>

    #include <Python.h>
    #include <rapidsmpf/error.hpp>

    namespace {

    /**
     * @brief Set Python exception from C++ exception
     *
     * Loads the specified Python exception class from the given module and sets
     * it as the current Python exception with the message from the C++ exception.
     *
     * @tparam ExceptT C++ exception type (must have a what() method)
     * @param exn The C++ exception object
     * @param module_name Python module name containing the exception class
     * @param class_name Python exception class name within the module
     */
    template<typename ExceptT>
    void set_exception(
      ExceptT const &exn, char const* module_name, char const* class_name
    ) {
        PyObject* module = PyImport_ImportModule(module_name);
        RAPIDSMPF_EXPECTS_FATAL(
          module != nullptr, "cannot find " + std::string{module_name}
        );
        PyObject* exc_class = PyObject_GetAttrString(module, class_name);
        RAPIDSMPF_EXPECTS_FATAL(
          exc_class != nullptr, "cannot find " + std::string{class_name}
        );
        PyErr_SetString(exc_class, exn.what());
        Py_DECREF(exc_class);
        Py_DECREF(module);
    }

    void ex_handler()
    {
      try {
        if (PyErr_Occurred())
          ;  // let latest Python exn pass through and ignore the current one
        throw;
      } catch (const rapidsmpf::reservation_error& exn) {
        set_exception(exn, "rapidsmpf.error", "ReservationError");
      } catch (const rapidsmpf::out_of_memory& exn) {
        set_exception(exn, "rapidsmpf.error", "OutOfMemory");
      } catch (const rapidsmpf::bad_alloc& exn) {
        set_exception(exn, "rapidsmpf.error", "BadAlloc");
      } catch (const std::bad_alloc& exn) {
        PyErr_SetString(PyExc_MemoryError, exn.what());
      } catch (const std::bad_cast& exn) {
        PyErr_SetString(PyExc_TypeError, exn.what());
      } catch (const std::domain_error& exn) {
        PyErr_SetString(PyExc_ValueError, exn.what());
      } catch (const std::invalid_argument& exn) {
        PyErr_SetString(PyExc_ValueError, exn.what());
      } catch (const std::ios_base::failure& exn) {
        PyErr_SetString(PyExc_IOError, exn.what());
      } catch (const std::out_of_range& exn) {
        PyErr_SetString(PyExc_IndexError, exn.what());
      } catch (const std::overflow_error& exn) {
        PyErr_SetString(PyExc_OverflowError, exn.what());
      } catch (const std::range_error& exn) {
        PyErr_SetString(PyExc_ArithmeticError, exn.what());
      } catch (const std::underflow_error& exn) {
        PyErr_SetString(PyExc_ArithmeticError, exn.what());
      } catch (const std::exception& exn) {
        PyErr_SetString(PyExc_RuntimeError, exn.what());
      } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Unknown exception");
      }
    }

    }  // anonymous namespace
    """
    cdef void ex_handler()
