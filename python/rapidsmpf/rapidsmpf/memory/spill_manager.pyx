# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from cython.operator cimport dereference as deref
from libc.stddef cimport size_t
from libc.stdint cimport uint64_t
from libcpp.optional cimport optional
from libcpp.vector cimport vector

from rapidsmpf._detail.exception_handling cimport (
    CppExcept, ex_handler, throw_py_as_cpp_exception,
    translate_py_to_cpp_exception)
from rapidsmpf.memory.buffer_resource cimport BufferResource

import weakref
from dataclasses import dataclass


@dataclass(frozen=True)
class SpillAttempt:
    event_id: int
    attempt_id: int
    start_ns: int
    requested_bytes: int
    reason: SpillReason
    is_background: bool
    evictor: int | None


@dataclass(frozen=True)
class SpillTransfer:
    event_id: int
    attempt_id: int
    submission_start_ns: int
    submission_end_ns: int
    submitted_bytes: int
    source: MemoryType
    destination: MemoryType
    reason: SpillReason
    is_background: bool
    evictor: int | None
    buffer_owner: int | None
    is_success: bool


cdef object optional_token_to_python(optional[uint64_t] token):
    if token.has_value():
        return token.value()
    return None


cdef size_t cython_invoke_python_spill_function(
    void* py_spill_function, size_t amount
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
            return (<object?>py_spill_function)(amount)
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
        return [wrapper, py_spill_function](std::size_t amount) -> std::size_t {
            return wrapper(py_spill_function, amount);
        };
    }
    """
    cpp_SpillFunction cython_to_cpp_closure_lambda(
         size_t (*wrapper)(void *, size_t),
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

    def add_spill_function(self, func, int priority, buffer_owner=None):
        """
        Adds a spill function with a given priority to the spill manager.

        The spill function is prioritized according to the specified priority value.

        Parameters
        ----------
        spill_function
            The spill function to be added.
        priority
            The priority level of the spill function (higher values indicate higher
            priority).
        buffer_owner
            Optional opaque process-local token identifying the owner of buffers
            managed by this spill function.

        Returns
        -------
        The ID assigned to the newly added spill function.
        """
        self._valid_buffer_resource()
        cdef size_t func_id
        cdef optional[uint64_t] cpp_buffer_owner
        if buffer_owner is not None:
            cpp_buffer_owner = <uint64_t>buffer_owner
        with nogil:
            func_id = deref(self._handle).add_spill_function(
                cython_to_cpp_closure_lambda(
                    cython_invoke_python_spill_function, <void *>func
                ),
                priority,
                cpp_buffer_owner
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

    def spill(self, size_t amount, SpillReason reason=SpillReason.EXPLICIT, evictor=None):
        """
        Initiates spilling to free up a specified amount of memory.

        This method iterates through registered spill functions in priority order,
        invoking them until at least the requested amount of memory has been spilled
        or no more spilling is possible.

        Parameters
        ----------
        amount
            The amount of memory (in bytes) to spill.
        reason
            The mutually exclusive reason for the spill attempt.
        evictor
            Optional opaque process-local token identifying the work requesting
            the spill.

        Returns
        -------
        The actual amount of memory spilled (in bytes), which may be more, less,
        or equal to the requested amount.
        """
        self._valid_buffer_resource()
        cdef size_t ret
        cdef optional[uint64_t] cpp_evictor
        if evictor is not None:
            cpp_evictor = <uint64_t>evictor
        with nogil:
            ret = deref(self._handle).spill(amount, reason, cpp_evictor)
        return ret

    def enable_event_collection(self, size_t capacity=4096):
        """Enable bounded spill-event collection and preallocate its storage."""
        self._valid_buffer_resource()
        with nogil:
            deref(self._handle).event_collector().enable(capacity)

    def disable_event_collection(self):
        """Disable collection and release stored events."""
        self._valid_buffer_resource()
        with nogil:
            deref(self._handle).event_collector().disable()

    @property
    def event_collection_enabled(self):
        self._valid_buffer_resource()
        with nogil:
            ret = deref(self._handle).event_collector().enabled()
        return ret

    @property
    def event_sequence(self):
        """Return the exclusive sequence cursor for events recorded so far."""
        self._valid_buffer_resource()
        cdef uint64_t ret
        with nogil:
            ret = deref(self._handle).event_collector().sequence()
        return ret

    @property
    def dropped_events(self):
        """Return the number of events overwritten or lost during collection."""
        self._valid_buffer_resource()
        cdef uint64_t ret
        with nogil:
            ret = deref(self._handle).event_collector().dropped_events()
        return ret

    def read_spill_attempts(self, uint64_t begin_sequence, end_sequence=None):
        """Read attempts in the half-open sequence range ``[begin, end)``."""
        self._valid_buffer_resource()
        cdef uint64_t end
        cdef vector[cpp_SpillAttemptRecord] records
        if end_sequence is None:
            end = deref(self._handle).event_collector().sequence()
        else:
            end = <uint64_t>end_sequence
        with nogil:
            records = deref(self._handle).event_collector().read_attempts(
                begin_sequence, end
            )
        return [
            SpillAttempt(
                record.event_id,
                record.attempt_id,
                record.start_ns,
                record.requested_bytes,
                record.reason,
                record.is_background,
                optional_token_to_python(record.evictor),
            )
            for record in records
        ]

    def read_spill_transfers(self, uint64_t begin_sequence, end_sequence=None):
        """Read copy submissions in the half-open sequence range ``[begin, end)``."""
        from rapidsmpf.memory.buffer import MemoryType as PyMemoryType

        self._valid_buffer_resource()
        cdef uint64_t end
        cdef vector[cpp_SpillTransferRecord] records
        if end_sequence is None:
            end = deref(self._handle).event_collector().sequence()
        else:
            end = <uint64_t>end_sequence
        with nogil:
            records = deref(self._handle).event_collector().read_transfers(
                begin_sequence, end
            )
        return [
            SpillTransfer(
                record.event_id,
                record.attempt_id,
                record.submission_start_ns,
                record.submission_end_ns,
                record.submitted_bytes,
                PyMemoryType(record.source),
                PyMemoryType(record.destination),
                record.reason,
                record.is_background,
                optional_token_to_python(record.evictor),
                optional_token_to_python(record.buffer_owner),
                record.is_success,
            )
            for record in records
        ]

    def spill_to_make_headroom(self, int64_t headroom = 0):
        """
        Attempts to free memory by spilling until the requested headroom is reservable.

        The headroom measurement is a snapshot, so a later ``reserve()`` of
        ``headroom`` bytes is not guaranteed to succeed. Spilling is performed
        in order of the function priorities until the requested headroom is
        reservable or no more spilling is possible. Spilling reduces
        allocations, never outstanding reservations.

        Parameters
        ----------
        headroom
            The target amount of headroom (in bytes). A negative headroom
            triggers spilling only once the memory available for reservation
            drops below ``headroom``.

        Returns
        -------
        The actual amount of memory spilled (in bytes), which may be less than
        requested if there is insufficient spillable data, but may also be more
        or equal to requested depending on the sizes of spillable data buffers.

        See Also
        --------
        BufferResource.memory_available_for_reservation
        """
        self._valid_buffer_resource()
        cdef size_t ret
        with nogil:
            ret = deref(self._handle).spill_to_make_headroom(headroom)
        return ret
