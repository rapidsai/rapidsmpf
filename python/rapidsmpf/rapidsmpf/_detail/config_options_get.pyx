# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cpython.ref cimport PyObject
from libc.stdint cimport int64_t
from libcpp cimport bool as bool_t
from libcpp.string cimport string

from rapidsmpf._detail.exception_handling cimport (
    CppExcept, ex_handler, throw_py_as_cpp_exception,
    translate_py_to_cpp_exception)
from rapidsmpf.config cimport cpp_Options


cdef extern from *:
    """
    template<typename T>
    T cython_to_cpp_closure_lambda(
        rapidsmpf::config::Options &options,
        std::string const& key,
        T (*wrapper)(void *, std::string),
        void* py_factory
    )
    {
        return options.get<T>(
            key,
            [wrapper, py_factory](std::string const &value) -> T {
                return wrapper(py_factory, value);
            }
        );
    }
    """
    T cython_to_cpp_closure_lambda[T](
        cpp_Options options,
        string key,
        T (*wrapper)(void *, string),
        void* py_factory
    ) except +ex_handler nogil


#########################################################################
# In the following, we implement `get` for each supported type.
# Since Cython doesn't support templated `cdef` functions, we simply
# implement all of them here. Note, we cannot use Cython's fused typed
# because we call the function from C++, which requires "real" templates.

cdef bool_t _invoke_factory_bool(
    void* py_factory, string value
) noexcept nogil:
    cdef CppExcept err
    with gil:
        try:
            return (<object?>py_factory)(value.decode("UTF-8"))
        except BaseException as e:
            err = translate_py_to_cpp_exception(e)
    throw_py_as_cpp_exception(err)


cdef get_bool(Options options, str key, factory):
    cdef string _key = str.encode(key)
    cdef bool_t _ret
    with nogil:
        _ret = cython_to_cpp_closure_lambda[bool_t](
            options._handle,
            _key,
            _invoke_factory_bool,
            <void *>factory,
        )
    return _ret


cdef int64_t _invoke_factory_int64(
    void* py_factory, string value
) noexcept nogil:
    cdef CppExcept err
    with gil:
        try:
            return (<object?>py_factory)(value.decode("UTF-8"))
        except BaseException as e:
            err = translate_py_to_cpp_exception(e)
    throw_py_as_cpp_exception(err)


cdef get_int(Options options, str key, factory):
    cdef string _key = str.encode(key)
    cdef int64_t _ret
    with nogil:
        _ret = cython_to_cpp_closure_lambda[int64_t](
            options._handle,
            _key,
            _invoke_factory_int64,
            <void *>factory,
        )
    return _ret


cdef double _invoke_factory_double(
    void* py_factory, string value
) noexcept nogil:
    cdef CppExcept err
    with gil:
        try:
            return (<object?>py_factory)(value.decode("UTF-8"))
        except BaseException as e:
            err = translate_py_to_cpp_exception(e)
    throw_py_as_cpp_exception(err)


cdef get_float(Options options, str key, factory):
    cdef string _key = str.encode(key)
    cdef double _ret
    with nogil:
        _ret = cython_to_cpp_closure_lambda[double](
            options._handle,
            _key,
            _invoke_factory_double,
            <void *>factory,
        )
    return _ret


cdef string _invoke_factory_string(
    void* py_factory, string value
) noexcept nogil:
    cdef CppExcept err
    with gil:
        try:
            return str.encode((<object?>py_factory)(value.decode("UTF-8")))
        except BaseException as e:
            err = translate_py_to_cpp_exception(e)
    throw_py_as_cpp_exception(err)


cdef get_str(Options options, str key, factory):
    cdef string _key = str.encode(key)
    cdef string _ret
    with nogil:
        _ret = cython_to_cpp_closure_lambda[string](
            options._handle,
            _key,
            _invoke_factory_string,
            <void *>factory,
        )
    return _ret.decode("UTF-8")


# For PyObject support, we implement `PyObjectSharedPtr`, which is
# a wrapper around a PyObject* that handles reference counting.
cdef extern from *:
    """
    #include <memory>
    #include <Python.h>
    struct PyDeleter {
        void operator()(PyObject* obj) const {
            if (obj) {
                // Acquire GIL before calling Py_DECREF
                PyGILState_STATE state = PyGILState_Ensure();
                Py_DECREF(obj);
                PyGILState_Release(state);
            }
        }
    };
    using PyObjectSharedPtr = std::shared_ptr<PyObject>;
    inline PyObjectSharedPtr cpp_make_shared_pyobject(PyObject* obj) {
        Py_INCREF(obj);
        return PyObjectSharedPtr(obj, PyDeleter());
    }
    """

    cdef cppclass PyObjectSharedPtr:
        PyObjectSharedPtr() except +ex_handler
        PyObject* get() noexcept

    PyObjectSharedPtr cpp_make_shared_pyobject(PyObject*)


cdef PyObjectSharedPtr _make_shared_pyobject(object obj):
    # Cython complains if we do this cast as an oneliner in
    # _invoke_factory_py_obj().
    return cpp_make_shared_pyobject(<PyObject*> obj)


cdef PyObjectSharedPtr _invoke_factory_py_obj(
    void* py_factory, string value
) noexcept nogil:
    cdef CppExcept err
    with gil:
        try:
            return _make_shared_pyobject((<object?>py_factory)(value.decode("UTF-8")))
        except BaseException as e:
            err = translate_py_to_cpp_exception(e)
    throw_py_as_cpp_exception(err)


cdef get_py_obj(Options options, str key, factory):
    cdef string _key = str.encode(key)
    cdef PyObjectSharedPtr ret
    with nogil:
        ret = cython_to_cpp_closure_lambda[PyObjectSharedPtr](
            options._handle,
            _key,
            _invoke_factory_py_obj,
            <void *>factory,
        )
    return <object?>ret.get()
