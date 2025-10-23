# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Minimal payload type for wrapping arbitrary Python objects in streaming messages.
"""

from cpython.object cimport PyObject
from cpython.ref cimport Py_DECREF, Py_INCREF, Py_XDECREF
from cython.operator cimport dereference as deref
from libc.stdint cimport uint64_t
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from rapidsmpf.streaming.core.channel cimport Message, cpp_Message


# Helper functions to release TypeErasedChunk from a message
cdef extern from *:
    """
    #include <cstdint>
    #include <memory>
    #include <rapidsmpf/streaming/cudf/owning_wrapper.hpp>

    namespace rapidsmpf::streaming {

    struct TypeErasedChunk {
        std::uint64_t sequence_{};
        OwningWrapper obj_{};

        TypeErasedChunk() = default;

        TypeErasedChunk(OwningWrapper&& obj, std::uint64_t sequence)
            : obj_(std::move(obj)), sequence_(sequence) {}

        // Move constructor
        TypeErasedChunk(TypeErasedChunk&& other) noexcept
            : obj_(std::move(other.obj_)), sequence_(other.sequence_) {}

        // Move assignment
        TypeErasedChunk& operator=(TypeErasedChunk&& other) noexcept {
            if (this != &other) {
                obj_ = std::move(other.obj_);
                sequence_ = other.sequence_;
            }
            return *this;
        }

        // Release the owned PyObject
        [[nodiscard]] void* release() { return obj_.release(); }

        // Delete copy constructor and copy assignment
        TypeErasedChunk(const TypeErasedChunk&) = delete;
        TypeErasedChunk& operator=(const TypeErasedChunk&) = delete;
    };

    }  // namespace rapidsmpf::streaming

    namespace {
    std::unique_ptr<rapidsmpf::streaming::TypeErasedChunk>
    cpp_release_from_message(rapidsmpf::streaming::Message&& msg) {
        return std::make_unique<rapidsmpf::streaming::TypeErasedChunk>(
            msg.release<rapidsmpf::streaming::TypeErasedChunk>()
        );
    }
    }  // anonymous namespace
    """
    cdef cppclass cpp_TypeErasedChunk "rapidsmpf::streaming::TypeErasedChunk":
        uint64_t sequence_
        cpp_OwningWrapper obj_  # Public member for accessing wrapped object
        cpp_TypeErasedChunk() except +
        cpp_TypeErasedChunk(cpp_OwningWrapper, uint64_t) except +
        void* release() except +

    cdef cppclass cpp_OwningWrapper "rapidsmpf::streaming::OwningWrapper":
        cpp_OwningWrapper() except +
        cpp_OwningWrapper(void*, void (*)(void*)) except +
        void* release() except +

    unique_ptr[cpp_TypeErasedChunk] cpp_release_from_message(cpp_Message) except +


cdef void py_deleter(void *p) noexcept nogil:
    with gil:
        Py_XDECREF(<PyObject*>p)


cdef class PyObjectPayload:
    """
    A payload that wraps an arbitrary Python object for streaming.

    This allows passing any Python object (dict, list, custom classes, etc.)
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
    >>> assert payload2.extract_object() == {"key1": 100, "key2": 200}
    """

    def __init__(self):
        raise ValueError(
            "use PyObjectPayload.from_object() or from_message()"
        )

    def __dealloc__(self):
        # The C++ destructor will handle cleanup via OwningWrapper
        self._handle.reset()

    @staticmethod
    cdef PyObjectPayload from_handle(unique_ptr[cpp_TypeErasedChunk] handle):
        """
        Construct a PyObjectPayload from an existing C++ handle.

        Parameters
        ----------
        handle
            A unique pointer to a C++ TypeErasedChunk.

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
        A new payload wrapping the given object.
        """
        # Increment reference count - will be decremented by py_deleter
        Py_INCREF(obj)
        cdef cpp_OwningWrapper wrapper = cpp_OwningWrapper(
            <void*><PyObject*>obj,
            py_deleter
        )
        cdef unique_ptr[cpp_TypeErasedChunk] chunk_ptr = unique_ptr[
            cpp_TypeErasedChunk
        ](new cpp_TypeErasedChunk(move(wrapper), sequence_number))
        return PyObjectPayload.from_handle(move(chunk_ptr))

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
            cpp_release_from_message(move(message._handle))
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

    cdef const cpp_TypeErasedChunk* handle_ptr(self):
        """
        Return a pointer to the underlying C++ TypeErasedChunk.

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

    cdef unique_ptr[cpp_TypeErasedChunk] release_handle(self):
        """
        Release ownership of the underlying C++ TypeErasedChunk.

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
        The sequence number.
        """
        return deref(self.handle_ptr()).sequence_

    def extract_object(self):
        """
        Extract and consume the wrapped Python object.

        This is a destructive operation - after calling this method, the payload
        is no longer usable and subsequent calls will raise an error.

        Returns
        -------
        The Python object that was wrapped by this payload.

        Raises
        ------
        ValueError
            If the payload has already been released.

        Warnings
        --------
        The payload is consumed by this operation and must not be used afterwards.
        """
        if not self._handle:
            raise ValueError("is uninitialized, has it been released?")
        cdef void* obj_ptr = self._handle.get().release()
        if obj_ptr == NULL:
            raise ValueError("Python object has already been released")
        cdef PyObject* py_obj = <PyObject*>obj_ptr
        # Cast to Python object.
        # Cython will increment refcount for the return value.
        cdef object result = <object>py_obj
        # Now decrement to balance the Py_INCREF from from_object().
        # The caller will own the one remaining reference.
        Py_DECREF(result)
        return result
