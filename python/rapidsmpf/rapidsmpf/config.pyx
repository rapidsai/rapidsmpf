# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cpython.bytes cimport PyBytes_FromStringAndSize
from cython.operator cimport dereference as deref
from cython.operator cimport preincrement as inc
from libc.stdint cimport int64_t
from libc.string cimport memcpy
from libcpp cimport bool as bool_t
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.utility cimport move
from libcpp.vector cimport vector

from rapidsmpf._detail cimport config_options_get

import os
import re


# Cython doesn't support a `std::function` type that uses a template argument,
# which is needed to declare `rapidsmpf::config::OptionFactory`. To handle this,
# we implement `cpp_options_get_using_parse_string()`, which calls `options.get<T>()`
# with a lambda function that:
#  - use `rapidsmpf::parse_string<T>()` to convert the option to type T, or
#  - return `default_value` if the option isn't found.
# TODO: implement a similar function for handle python objects.
cdef extern from *:
    """
    #include <rapidsmpf/utils.hpp>
    template<typename T>
    T cpp_options_get_using_parse_string(
        rapidsmpf::config::Options &options, std::string const& key, T default_value
    )
    {
        return options.get<T>(
            key,
            [default_value = std::move(default_value)](std::string const&x)
            {
                if(x.empty()) {
                    return std::move(default_value);
                }
                return rapidsmpf::parse_string<T>(x);
            }
        );
    }
    """
    T cpp_options_get_using_parse_string[T](
        cpp_Options options, string key, T default_value
    ) except + nogil


# Supported types for use with _options_get_using_parse_string().
ctypedef fused DefaultValueT:
    bool_t
    int64_t
    double
    string

cdef DefaultValueT _options_get_using_parse_string(
    cpp_Options options, string key, DefaultValueT default_value
):
    """Release the GIL and retrieve an option value."""
    cdef DefaultValueT ret
    with nogil:
        ret = cpp_options_get_using_parse_string(
            options, key, default_value
        )
    return ret


cdef class Options:
    """Initialize an Options object with a dictionary of string options.

    Parameters
    ----------
    options_as_strings
        A dictionary representing option names and their corresponding values.
    """
    def __cinit__(self, options_as_strings = None):
        cdef unordered_map[string, string] opts
        if options_as_strings is not None:
            for key, val in options_as_strings.items():
                opts[str.encode(key)] = str.encode(val)
        with nogil:
            self._handle = cpp_Options(move(opts))

    def __dealloc__(self):
        with nogil:
            self._handle = cpp_Options()

    def get(self, str key, *, return_type, factory):
        """
        Retrieves a configuration option by key.

        If the option is not present, it is constructed using the provided
        factory function, which receives the string representation of the
        option (or an empty string if unset). The option is cached after
        the first access.

        The option is cast to the specified `return_type`, which must be one
        of the supported primitive types: `bool`, `int`, `float`, or `str`.

        Once a key has been accessed with a particular `return_type`, subsequent
        calls to `get` with the same key must use the same `return_type`.
        Using a different type for the same key will result in a `TypeError`.

        Parameters
        ----------
        key
            The option key. Should be in lowercase.
        return_type
            The return type. Must be one of: `bool`, `int`, `float`, `str`.
        factory
            A factory function that constructs an instance of the desired type
            from a string representation.

        Returns
        -------
        The value of the requested option, cast to the specified `return_type`.

        Raises
        ------
        ValueError
            If the `return_type` is unsupported, or if the stored option type
            does not match the expected type.
        TypeError
            If the option has already been accessed with a different `return_type`.

        Warning
        -------
        The factory must not access the `Options` instance, as this may lead
        to a deadlock due to internal locking.

        Notes
        -----
        - This function dispatches internally to type-specific getters.
        - Support for custom Python object return types (`PyObject`) is not yet
          implemented.
        - Only the following types are currently supported: `bool`, `int`, `float`,
          `str`.
        """
        if issubclass(return_type, bool):
            return config_options_get.get_bool(self, key, factory)
        elif issubclass(return_type, int):
            return config_options_get.get_int(self, key, factory)
        elif issubclass(return_type, float):
            return config_options_get.get_float(self, key, factory)
        elif issubclass(return_type, str):
            return config_options_get.get_str(self, key, factory)

        # TODO: handle PyObject return type.
        raise ValueError(
            f"return type ({type(return_type)}) is not supported, "
            r"supported types: {bool, int, float, str}."
        )

    def get_strings(self):
        """
        Get all option key-value pairs as strings.

        Returns
        -------
        A dictionary containing all stored options, where the keys and values are
        both strings.
        """
        cdef unordered_map[string, string] strings
        with nogil:
            strings = self._handle.get_strings()
        cdef dict ret = {}
        cdef unordered_map[string, string].iterator it = strings.begin()
        while it != strings.end():
            k = deref(it).first.decode("utf-8")
            v = deref(it).second.decode("utf-8")
            ret[k] = v
            inc(it)
        return ret

    def serialize(self) -> bytes:
        """
        Serialize the `Options` object into a binary buffer.

        This method produces a compact binary representation of the internal
        key-value options. The format is suitable for storage or transmission
        and can be later restored using `Options.deserialize()`.

        The binary format is:
            - [uint64_t count] — number of key-value pairs
            - [count * 2 * uint64_t] — offset pairs (key_offset, value_offset)
            - [raw bytes] — key and value strings stored contiguously

        Notes
        -----
        An Options instance can only be serialized if no options have been
        accessed. This is because serialization is based on the original
        string representations of the options. Once an option has been
        accessed and parsed, its string value may no longer accurately
        reflect its state, making serialization potentially inconsistent.

        Returns
        -------
        bytes
            A `bytes` object containing the serialized binary representation
            of the options.

        Raises
        ------
        ValueError
            If any option has already been accessed and cannot be serialized.
        """
        cdef vector[uint8_t] vec
        with nogil:
            vec = self._handle.serialize()
        assert vec.size() > 0, "C++ serialize result corrupted"
        return <bytes>PyBytes_FromStringAndSize(<const char*>&vec[0], vec.size())

    @staticmethod
    def deserialize(bytes serialized_buffer):
        """
        Deserialize a binary buffer into an `Options` object.

        This method reconstructs an `Options` instance from a byte buffer
        produced by the `Options.serialize()` method.

        See `Options.serialize()` for the binary format.

        Parameters
        ----------
        serialized_buffer
            A buffer containing serialized options in the defined binary format.

        Returns
        -------
        Options
            A reconstructed `Options` instance containing the deserialized key-value
            pairs.

        Raises
        ------
        ValueError
            If the input buffer is malformed or inconsistent with the expected format.
        """
        cdef Py_ssize_t size = len(serialized_buffer)
        cdef const char* src = <const char*>serialized_buffer
        cdef vector[uint8_t] vec
        cdef Options ret = Options.__new__(Options)
        with nogil:
            vec.resize(size)
            memcpy(<void*>vec.data(), src, size)
            ret._handle = cpp_Options.deserialize(vec)
        return ret

    def __getstate__(self):
        """
        Get the state of the object for pickling.

        This method is called by the `pickle` module to retrieve a serialized
        representation of the `Options` object. It uses the `serialize()`
        method to return the internal state as a `bytes` object.

        Returns
        -------
        A binary representation of the `Options` object, suitable for pickling.
        """
        return self.serialize()

    def __setstate__(self, state):
        """
        Set the state of the object during unpickling.

        This method is called by the `pickle` module to restore the object's state
        from a serialized `bytes` buffer. It uses the `deserialize()` method and
        assigns the resulting internal handle.

        Parameters
        ----------
        state
            A `bytes` object representing the serialized state of the `Options` object.
        """
        cdef Options options = self.deserialize(state)
        self._handle = options._handle


def get_environment_variables(str key_regex = "RAPIDSMPF_(.*)"):
    """
    Returns a dictionary of environment variables matching a given regular expression.

    This function scans the current process's environment variables and inserts those
    whose keys match the provided regular expression. The regular expression must
    contain exactly one capture group to extract the portion of the environment variable
    key to use as the dictionary key.

    For example, to strip the `RAPIDSMPF_` prefix, use `r"RAPIDSMPF_(.*)"` as the regex.
    The captured group will be used as the key in the output dictionary.

    Example:
        - Environment variable: RAPIDSMPF_FOO=bar
        - key_regex: r"RAPIDSMPF_(.*)"
        - Resulting dictionary entry: { "FOO": "bar" }

    Parameters
    ----------
    key_regex
        A regular expression with a single capture group to match and extract
        the environment variable keys.

    Returns
    -------
    A dictionary containing all matching environment variables, with keys as
    extracted by the capture group.

    Raises
    ------
    ValueError
        If `key_regex` does not contain exactly one capture group.

    See Also
    --------
    os.environ : Dictionary of the current environment variables.
    """
    pattern = re.compile(key_regex)
    if pattern.groups != 1:
        raise ValueError(
            "key_regex must contain exactly one capture group (e.g., 'RAPIDSMPF_(.*)')"
        )
    ret = {}
    for key, value in os.environ.items():
        match = pattern.fullmatch(key)
        if match:
            ret[match.group(1)] = value
    return ret
