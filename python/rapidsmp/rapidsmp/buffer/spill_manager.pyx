# Copyright (c) 2025, NVIDIA CORPORATION.

from cython.operator cimport dereference as deref
from libc.stddef cimport size_t
from libcpp.pair cimport pair
from libcpp.string cimport string
from rapidsmp.buffer.resource cimport BufferResource

# Transparent handle of a C++ exception
ctypedef pair[int, string] CppExcept

cdef CppExcept translate_py_to_cpp_exception(py_exception) noexcept:
    """Translate a Python exception into a C++ exception handle (`CppExcept`).

    The returned exception handle can then be thrown by `throw_py_as_cpp_exception()`,
    which MUST be done without holding the GIL.

    This is useful when C++ calls a Python function and needs to catch or
    propagate exceptions.
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

# Implementation of `throw_py_as_cpp_exception()`, which throws a given `CppExcept`.
# This function MUST be called without the GIL otherwise the thrown C++ exception
# are translated back into a Python exception.
cdef extern from *:
    """
    #include <ios>
    void throw_py_as_cpp_exception(std::pair<int, std::string> const &res) {
        switch(res.first) {
            case 0:
                throw rmm::out_of_memory(res.second);
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
    void throw_py_as_cpp_exception(CppExcept) nogil

cdef size_t cython_invoke_python_spill_function(
    void* py_spill_function, size_t amount
) noexcept nogil:
    """
    # Note that this function is designed to rethrow Python exceptions as
    # C++ exceptions when called as a callback from C++, so it is noexcept
    # from Cython's perspective.
    """
    cdef CppExcept err
    with gil:
        try:
            return (<object>py_spill_function)(amount)
        except BaseException as e:
            err = translate_py_to_cpp_exception(e)
    throw_py_as_cpp_exception(err)

cdef extern from *:
    """
    template<typename T1, typename T2>
    rapidsmp::SpillManager::SpillFunction cython_to_cpp_closure_lambda(
        T1 wrapper, T2 py_spill_function
    ) {
        return [wrapper, py_spill_function](std::size_t amount) -> std::size_t {
            return wrapper(py_spill_function, amount);
        };
    }
    """
    cpp_SpillFunction cython_to_cpp_closure_lambda(
         size_t (*wrapper)(void *, size_t),
         void *py_spill_function
    ) nogil

cdef class SpillManager:
    """
    Class manages memory spilling to free up device memory when needed.

    The SpillManager is responsible for registering, prioritizing, and executing
    spill functions to ensure efficient memory management.
    """
    def __init__(self):
        raise TypeError("Please get a `SpillManager` from a buffer resource instance")

    @classmethod
    def _create(cls, BufferResource br):
        cdef SpillManager ret = cls.__new__(cls)
        ret._handle = &(deref(br._handle).cpp_spill_manager())
        ret._owner = br
        ret._spill_functions = {}
        return ret

    def add_spill_function(self, func, int priority):
        cdef size_t func_id = deref(self._handle).add_spill_function(
            cython_to_cpp_closure_lambda(
                cython_invoke_python_spill_function, <void *>func
            ),
            priority
        )
        self._spill_functions[func_id] = func
        return func_id

    def remove_spill_function(self, int function_id):
        deref(self._handle).remove_spill_function(function_id)

    def spill(self, size_t amount):
        return deref(self._handle).spill(amount)
