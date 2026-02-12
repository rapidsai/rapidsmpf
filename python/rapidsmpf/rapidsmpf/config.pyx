# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cpython.bytes cimport PyBytes_FromStringAndSize
from cython.operator cimport dereference as deref
from cython.operator cimport preincrement
from libc.string cimport memcpy
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.utility cimport move
from libcpp.vector cimport vector

from rapidsmpf._detail cimport config_options_get

import os
import re

from rapidsmpf.utils.string import parse_boolean, parse_bytes


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

    def insert_if_absent(self, dict options_as_strings):
        """Insert multiple options if they are not already present.

        Attempts to insert each key-value pair from the provided dictionary,
        skipping keys that already exist in the options.

        Parameters
        ----------
        options_as_strings
            Dictionary of option keys mapped to their string representations.
            Keys are inserted only if they do not already exist. The keys are
            trimmed and converted to lower case before insertion.

        Returns
        -------
        Number of newly inserted options (0 if none were added).
        """
        cdef unordered_map[string, string] opts
        for key, val in options_as_strings.items():
            opts[str.encode(key)] = str.encode(val)
        cdef size_t ret
        with nogil:
            ret = self._handle.insert_if_absent(move(opts))
        return ret

    def get(self, str key not None, *, return_type, factory):
        """
        Retrieves a configuration option by key.

        If the option is not present, it is constructed using the provided
        factory function, which receives the string representation of the
        option (or an empty string if unset). The option is cached after
        the first access.

        The option is cast to the specified ``return_type``. To be accessible
        from C++, it must be one of: `bool`, `int`, `float`, `str`. Otherwise, it
        is stored as a ``PyObject*``.

        Once a key has been accessed with a particular ``return_type``, subsequent
        calls to `get` with the same key must use the same ``return_type``.
        Using a different type for the same key will result in a `TypeError`.

        Parameters
        ----------
        key
            The option key. Should be in lowercase.
        return_type
            The return type. To be accessible from C++, it must be one of: `bool`,
            `int`, `float`, `str`. Use `object` to indicate any Python type.
        factory
            A factory function that constructs an instance of the desired type
            from a string representation.

        Returns
        -------
        The value of the requested option, cast to the specified ``return_type``.

        Raises
        ------
        ValueError
            If the ``return_type`` is unsupported, or if the stored option type
            does not match the expected type.
        TypeError
            If the option has already been accessed with a different ``return_type``.

        Warnings
        --------
        The factory must not access the Options instance, as this may lead
        to a deadlock due to internal locking.
        """
        if issubclass(return_type, bool):
            return config_options_get.get_bool(self, key, factory)
        elif issubclass(return_type, int):
            return config_options_get.get_int(self, key, factory)
        elif issubclass(return_type, float):
            return config_options_get.get_float(self, key, factory)
        elif issubclass(return_type, str):
            return config_options_get.get_str(self, key, factory)
        return config_options_get.get_py_obj(self, key, factory)

    def get_or_default(self, str key not None, *, default_value):
        """
        Retrieve a configuration option by key, using a default value if not present.

        This is a convenience wrapper around `get()` that uses the type of the
        ``default_value`` as the return type and provides a default factory that
        parses a string into that type.

        Parameters
        ----------
        key
            The name of the option to retrieve.
        default_value
            The default value to return if the option is not set. Its type is used
            to determine the expected return type.

        Returns
        -------
        The value of the option if it exists and can be parsed to the type of
        ``default_value``, otherwise ``default_value``.

        Raises
        ------
        ValueError
            If the stored option value cannot be parsed to the required type.
        TypeError
            If the option has already been accessed with a different return type.

        Notes
        -----
        - This method infers the return type from ``type(default_value)``.
        - If ``default_value`` is used, it will be cached and reused for subsequent
          accesses of the same key.

        Examples
        --------
        >>> opts = Options()
        >>> opts.get_or_default("debug", default_value=False)
        False
        >>> opts.get_or_default("timeout", default_value=1.5)
        1.5
        >>> opts.get_or_default("level", default_value="info")
        'info'
        """
        if isinstance(default_value, bool):
            def factory(option_as_string):
                if option_as_string:
                    return parse_boolean(option_as_string)
                return default_value
        else:
            def factory(option_as_string):
                if option_as_string:
                    return type(default_value)(option_as_string)
                return default_value
        return self.get(key, return_type=type(default_value), factory=factory)

    def get_strings(self):
        """
        Get all option key-value pairs as strings.

        Options that do not have a string representation, such as options inserted
        as typed values in C++ are included with an empty string value.

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
            preincrement(it)
        return ret

    def serialize(self) -> bytes:
        """
        Serialize the Options object into a binary buffer.

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
    def deserialize(bytes serialized_buffer not None):
        """
        Deserialize a binary buffer into an Options object.

        This method reconstructs an Options instance from a byte buffer
        produced by the `Options.serialize()` method.

        See `Options.serialize()` for the binary format.

        Parameters
        ----------
        serialized_buffer
            A buffer containing serialized options in the defined binary format.

        Returns
        -------
        Options
            A reconstructed Options instance containing the deserialized key-value
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
        representation of the Options object. It uses the `serialize()`
        method to return the internal state as a `bytes` object.

        Returns
        -------
        A binary representation of the Options object, suitable for pickling.
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
            A `bytes` object representing the serialized state of the Options object.
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

    For example, to strip the ``RAPIDSMPF_`` prefix, use ``r"RAPIDSMPF_(.*)"`` as
    the regex. The captured group will be used as the key in the output dictionary.

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
        If ``key_regex`` does not contain exactly one capture group.

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


class Optional:
    """
    Represents an option value that can be explicitly disabled.

    This class wraps an option value and interprets certain strings as
    indicators that the value is disabled (case-insensitive): {"false", "no",
    "off", "disable", "disabled"}.

    This is typically used to simplify optional or Optional options with
    `Options.get_or_default()`.

    Parameters
    ----------
    value
        The input value to interpret.

    Attributes
    ----------
    value
        The raw input value, unless it matched a disable keyword, in which case
        the value is `None`.

    Examples
    --------
    >>> from rapidsmpf.config import Optional, Options
    >>> Optional("OFF").value
    None

    >>> Optional("no").value
    None

    >>> Optional("100").value
    '100'

    >>> Optional("").value
    ''

    >>> opts = Options()
    >>> opts.get_or_default(
    ...     "dask_periodic_spill_check",
    ...     default_value=Optional(1e-3)
    ... ).value
    0.001
    """

    def __init__(self, value):
        if str(value).strip().lower() in {"false", "no", "off", "disable", "disabled"}:
            self.value = None
        else:
            self.value = value


class OptionalBytes(Optional):
    """
    Represents a byte-sized option that can be explicitly disabled.

    This class is a specialization of `Optional` that interprets the input
    as a human-readable byte size string (e.g., "100 MB", "1KiB", "1e6").
    If the input is one of the disable keywords (e.g., "off", "no", "false"),
    the value is treated as disabled (`None`). Otherwise, it is parsed to an
    integer number of bytes using ``rapidsmpf.utils.string.parse_bytes()``.

    This is useful for configuration options that may be set to a size limit
    or explicitly turned off.

    Parameters
    ----------
    value
        A human-readable byte size (e.g., "1MiB", "100 MB") or a disable
        keyword (case-insensitive), or an integer number of bytes.

    Attributes
    ----------
    value
        The size in bytes, or `None` if disabled.

    Examples
    --------
    >>> from rapidsmpf.config import OptionalBytes
    >>> OptionalBytes("1KiB").value
    1024

    >>> OptionalBytes("OFF").value is None
    True

    >>> OptionalBytes(2048).value
    2048
    """
    def __init__(self, value):
        super().__init__(parse_bytes(value))
