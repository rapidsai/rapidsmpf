# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference as deref
from libc.stddef cimport size_t

from rapidsmpf._detail.exception_handling cimport (
    CppExcept, ex_handler, throw_py_as_cpp_exception,
    translate_py_to_cpp_exception)
from rapidsmpf.memory.buffer cimport MemoryType
from rapidsmpf.memory.buffer_resource cimport BufferResource

import weakref


cdef size_t cython_invoke_python_spill_function(
    void* py_spill_function, MemoryType mem_type, size_t amount
) noexcept nogil:
    """
    Invokes a Python spill function from C++ in a Cython-safe manner.

    This function calls a Python spill function while ensuring proper exception
    handling. If a Python exception occurs, it is translated into a corresponding
    C++ exception.

    Notice, we use the `noexcept` keyword to make sure Cython doesn't translate the
    C++ function back into a Python function.

    Parameters
    ----------
    py_spill_function
        A pointer to a Python callable that implements a spill function.
    mem_type
        The memory type to spill from.
    amount
        The amount of memory (in bytes) to be spilled.

    Returns
    -------
    The amount of memory actually spilled, as returned by the Python function.

    Raises
    ------
    Converts Python exceptions to C++ exceptions using `throw_py_as_cpp_exception`.
    """
    cdef CppExcept err
    with gil:
        try:
            return (<object?>py_spill_function)(mem_type, amount)
        except BaseException as e:
            err = translate_py_to_cpp_exception(e)
    throw_py_as_cpp_exception(err)

# To run a Python spill function in C++ with catchable exceptions, we use this function
# to create a lambda function that uses `cython_invoke_python_spill_function` to run
# the Python. The returned lambda function can then be given to the C++ SpillManager.
cdef extern from *:
    """
    template<typename T1, typename T2>
    rapidsmpf::SpillManager::SpillFunction cython_to_cpp_closure_lambda(
        T1 wrapper, T2 py_spill_function
    ) {
        return [wrapper, py_spill_function](
            rapidsmpf::MemoryType mem_type, std::size_t amount
        ) -> std::size_t {
            return wrapper(py_spill_function, mem_type, amount);
        };
    }
    """
    cpp_SpillFunction cython_to_cpp_closure_lambda(
         size_t (*wrapper)(void *, MemoryType, size_t),
         void *py_spill_function
    ) except +ex_handler nogil

cdef class SpillManager:
    """
    Class manages memory spilling to free up device memory when needed.

    The SpillManager is responsible for registering, prioritizing, and executing
    spill functions to ensure efficient memory management.
    """
    def __init__(self):
        raise TypeError("Please get a `SpillManager` from a buffer resource instance")

    def __dealloc__(self):
        """
        Deallocate resource without holding the GIL.

        This is important to ensure owned resources, like the underlying C++
        `SpillManager` object is destroyed, ensuring any threads can be
        joined without risk of deadlocks if both thread compete for the GIL.
        """
        with nogil:
            self._handle = NULL

    @classmethod
    def _create(cls, BufferResource br not None):
        """Construct a SpillManager associated the specified buffer resource.

        This shouldn't be used directly instead use `BufferResource.spill_manage)`.

        Parameters
        ----------
        br
            The associated buffer resource.

        Returns
        -------
        The new spill manager instance.
        """
        cdef SpillManager ret = cls.__new__(cls)
        with nogil:
            ret._handle = &(deref(br._handle).spill_manager())
        ret._br = weakref.ref(br)
        ret._spill_functions = {}
        return ret

    def _valid_buffer_resource(self):
        """Raise if the buffer resource has been deleted."""
        if self._br() is None:
            raise ValueError("The BufferResource must outlive the spill manager")

    def add_spill_function(self, func, int priority):
        """
        Adds a spill function with a given priority to the spill manager.

        The spill function is prioritized according to the specified priority value.

        Parameters
        ----------
        func
            The spill function to be added. It is called as ``func(mem_type, amount)``
            where ``mem_type`` is the :class:`~.MemoryType` to free and ``amount`` the
            requested number of bytes. It returns the number of bytes spilled, or zero
            if it holds no data in ``mem_type``.
        priority
            The priority level of the spill function (higher values indicate higher
            priority).

        Returns
        -------
        The ID assigned to the newly added spill function.
        """
        self._valid_buffer_resource()
        cdef size_t func_id
        with nogil:
            func_id = deref(self._handle).add_spill_function(
                cython_to_cpp_closure_lambda(
                    cython_invoke_python_spill_function, <void *>func
                ),
                priority
            )
        self._spill_functions[func_id] = func
        return func_id

    def remove_spill_function(self, int function_id):
        """
        Removes a spill function from the spill manager.

        This method unregisters the spill function associated with the given ID and
        removes it from the priority list. If no more spill functions remain, the
        periodic spill thread is paused.

        Parameters
        ----------
        The ID of the spill function to be removed.
        """
        self._valid_buffer_resource()
        with nogil:
            deref(self._handle).remove_spill_function(function_id)
        del self._spill_functions[function_id]

    def spill(self, size_t amount, mem_type=None):
        """
        Initiates spilling to free up a specified amount of memory.

        This method iterates through registered spill functions in priority order,
        invoking them until at least the requested amount of memory has been spilled
        or no more spilling is possible.

        Parameters
        ----------
        amount
            The amount of memory (in bytes) to spill.
        mem_type
            The memory type to spill from, ``MemoryType.DEVICE`` if not given.

        Raises
        ------
        ValueError
            If ``mem_type`` cannot be spilled, e.g. ``MemoryType.DISK``.

        Returns
        -------
        The actual amount of memory spilled (in bytes), which may be more, less,
        or equal to the requested amount.
        """
        self._valid_buffer_resource()
        cdef MemoryType mt = (
            MemoryType.DEVICE if mem_type is None else <MemoryType?>mem_type
        )
        cdef size_t ret
        with nogil:
            ret = deref(self._handle).spill(amount, mt)
        return ret
