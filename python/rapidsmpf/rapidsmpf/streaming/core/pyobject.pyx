# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Minimal payload type for wrapping arbitrary Python objects in streaming messages.
"""

from cpython.object cimport PyObject
from cpython.ref cimport Py_INCREF, Py_XDECREF
from cython.operator cimport dereference as deref
from libc.stdint cimport uint64_t
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.utility cimport move

from rapidsmpf.streaming.core.channel cimport Message, cpp_Message


# Define a minimal C++ struct inline to hold a Python object
cdef extern from *:
    """
    #include <cstdint>
    #include <memory>

    namespace rapidsmpf {
    namespace streaming {

    struct PyObjectPayload {
        std::uint64_t sequence_number;
        PyObject* py_obj;

        PyObjectPayload(std::uint64_t seq, PyObject* obj)
            : sequence_number(seq), py_obj(obj) {}

        // Move constructor
        PyObjectPayload(PyObjectPayload&& other) noexcept
            : sequence_number(other.sequence_number), py_obj(other.py_obj) {
            other.py_obj = nullptr;
        }

        // Move assignment
        PyObjectPayload& operator=(PyObjectPayload&& other) noexcept {
            if (this != &other) {
                sequence_number = other.sequence_number;
                py_obj = other.py_obj;
                other.py_obj = nullptr;
            }
            return *this;
        }

        // Delete copy constructor and copy assignment
        PyObjectPayload(const PyObjectPayload&) = delete;
        PyObjectPayload& operator=(const PyObjectPayload&) = delete;
    };

    }  // namespace streaming
    }  // namespace rapidsmpf

    // Helper function to release PyObjectPayload from a message
    namespace {
    std::unique_ptr<rapidsmpf::streaming::PyObjectPayload>
    cpp_release_pyobject_from_message(
        rapidsmpf::streaming::Message &&msg
    ) {
        return std::make_unique<rapidsmpf::streaming::PyObjectPayload>(
            msg.release<rapidsmpf::streaming::PyObjectPayload>()
        );
    }
    }
    """
    cdef cppclass cpp_PyObjectPayload "rapidsmpf::streaming::PyObjectPayload":
        uint64_t sequence_number
        PyObject* py_obj
        cpp_PyObjectPayload(uint64_t seq, PyObject* obj) except +

    unique_ptr[cpp_PyObjectPayload] \
        cpp_release_pyobject_from_message(cpp_Message) except +


cdef class PyObjectPayload:
    """
    A payload that wraps an arbitrary Python object for streaming.

    This allows passing any picklable Python object (dict, list, custom classes, etc.)
    through streaming channels without needing to convert to cuDF tables.

    Examples
    --------
    >>> from rapidsmpf.streaming.core.channel import Channel, Message
    >>> from rapidsmpf.streaming.core.pyobject import PyObjectPayload
    >>>
    >>> # Create a payload with a Python dict
    >>> data = {"key1": 100, "key2": 200}
    >>> payload = PyObjectPayload.from_object(sequence_number=0, obj=data)
    >>>
    >>> # Wrap in a message
    >>> msg = Message(payload)
    >>>
    >>> # Later, extract it back
    >>> payload2 = PyObjectPayload.from_message(msg)
    >>> assert payload2.get_object() == {"key1": 100, "key2": 200}
    """

    def __init__(self):
        raise ValueError(
            "use PyObjectPayload.from_object() or from_message()"
        )

    def __dealloc__(self):
        if self._handle:
            # Decrement the reference count when the payload is destroyed
            Py_XDECREF(deref(self._handle).py_obj)
        self._handle.reset()

    @staticmethod
    cdef PyObjectPayload from_handle(unique_ptr[cpp_PyObjectPayload] handle):
        """
        Construct a PyObjectPayload from an existing C++ handle.

        Parameters
        ----------
        handle
            A unique pointer to a C++ PyObjectPayload.

        Returns
        -------
        A new PyObjectPayload wrapping the given handle.
        """
        cdef PyObjectPayload ret = PyObjectPayload.__new__(PyObjectPayload)
        ret._handle = move(handle)
        return ret

    @staticmethod
    def from_object(uint64_t sequence_number, obj):
        """
        Create a PyObjectPayload from a Python object.

        Parameters
        ----------
        sequence_number
            Sequence number for this payload.
        obj
            Any Python object to wrap. The object's reference count is incremented
            and will be decremented when the payload is destroyed.

        Returns
        -------
        PyObjectPayload
            A new payload wrapping the given object.
        """
        # Increment reference count - will be decremented in __dealloc__
        Py_INCREF(obj)
        return PyObjectPayload.from_handle(
            make_unique[cpp_PyObjectPayload](
                sequence_number,
                <PyObject*>obj
            )
        )

    @staticmethod
    def from_message(Message message not None):
        """
        Construct a PyObjectPayload by consuming a Message.

        Parameters
        ----------
        message
            Message containing a PyObjectPayload. The message is released
            and is empty after this call.

        Returns
        -------
        A new PyObjectPayload extracted from the given message.
        """
        return PyObjectPayload.from_handle(
            cpp_release_pyobject_from_message(move(message._handle))
        )

    def into_message(self, Message message not None):
        """
        Move this PyObjectPayload into a Message.

        This method is not typically called directly. Instead, it is invoked by
        `Message.__init__()` when creating a new Message with this PyObjectPayload
        as its payload.

        Parameters
        ----------
        message
            Message object that will take ownership of this PyObjectPayload.

        Raises
        ------
        ValueError
            If the provided message is not empty.

        Warnings
        --------
        The PyObjectPayload is released and must not be used after this call.
        """
        if not message.empty():
            raise ValueError("cannot move into a non-empty message")
        message._handle = cpp_Message(self.release_handle())

    cdef const cpp_PyObjectPayload* handle_ptr(self):
        """
        Return a pointer to the underlying C++ PyObjectPayload.

        Returns
        -------
        Raw pointer to the underlying C++ object.

        Raises
        ------
        ValueError
            If the PyObjectPayload is uninitialized.
        """
        if not self._handle:
            raise ValueError("is uninitialized, has it been released?")
        return self._handle.get()

    cdef unique_ptr[cpp_PyObjectPayload] release_handle(self):
        """
        Release ownership of the underlying C++ PyObjectPayload.

        After this call, the current object is in a moved-from state and
        must not be accessed.

        Returns
        -------
        Unique pointer to the underlying C++ object.

        Raises
        ------
        ValueError
            If the PyObjectPayload is uninitialized.
        """
        if not self._handle:
            raise ValueError("is uninitialized, has it been released?")
        return move(self._handle)

    @property
    def sequence_number(self):
        """
        Return the sequence number of this payload.

        Returns
        -------
        int
            The sequence number.
        """
        return deref(self.handle_ptr()).sequence_number

    def get_object(self):
        """
        Get the wrapped Python object.

        Returns
        -------
        The Python object wrapped by this payload.

        Raises
        ------
        ValueError
            If the payload has been released.
        """
        cdef const cpp_PyObjectPayload* ptr = self.handle_ptr()
        if ptr.py_obj == NULL:
            raise ValueError("Python object has been released")
        return <object>ptr.py_obj
