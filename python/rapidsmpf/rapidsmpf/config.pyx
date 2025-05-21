# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int64_t
from libcpp cimport bool as bool_t
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.utility cimport move

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
    def __cinit__(self, options_as_strings):
        cdef unordered_map[string, string] opts
        for key, val in options_as_strings.items():
            opts[str.encode(key)] = str.encode(val)
        with nogil:
            self._handle = cpp_Options(move(opts))

    def __dealloc__(self):
        with nogil:
            self._handle = None

    def get_or_assign(self, str key, parser_type, default_value):
        """
        Get the value of the given key, or assign and return a default value.

        When a key is accessed for the first time, its option value (as a string)
        is parsed using the `parser_type`. If the key has no option value,
        `default_value` is assigned to the key and returned.

        The type of the returned value is determined by the `parser_type` argument,
        which must be one of the supported Python types: bool, int, float, or str.
        The `default_value` is cast to the corresponding C++ type before being passed
        to the underlying options system.

        TODO: Implement Python bindings to `rapidsmpf::config::Options::get` that
        support Python objects. This will make it possible to store and retrieve any
        kind of Python objects, not just bool, int, float, and str. The downside is
        that those Python objects will not be accessible to C++.

        Note
        ----
        Once a key has been accessed with a particular `parser_type`, subsequent
        calls to `get_or_assign` on the same key must use the same `parser_type`.
        Using a different `parser_type` for the same key will result in a `ValueError`.
        The first parsing determines the value type associated with that key.

        Parameters
        ----------
        key
            The key to look up or assign in the configuration.
        parser_type
            The expected type of the result. Must be one of: bool, int, float, str.
        default_value
            The value to assign and return if the key is not found. Its type must
            match `parser_type`.

        Returns
        -------
        The value associated with the key, parsed and cast to the given type,
        or the assigned default value if the key did not exist.

        Raises
        ------
        ValueError
            If the type of `default_value` is not supported or does not match
            `parser_type`.
        """
        if issubclass(parser_type, bool):
            return _options_get_using_parse_string[bool_t](
                self._handle, str.encode(key), bool(default_value)
            )
        elif issubclass(parser_type, int):
            return _options_get_using_parse_string[int64_t](
                self._handle, str.encode(key), int(default_value)
            )
        elif issubclass(parser_type, float):
            return _options_get_using_parse_string[double](
                self._handle, str.encode(key), float(default_value)
            )
        elif issubclass(parser_type, str):
            return _options_get_using_parse_string[string](
                self._handle, str.encode(key), str.encode(str(default_value))
            ).decode('UTF-8')
        raise ValueError(
            f"default type ({type(default_value)}) is not supported, "
            "please use `.get()` (not implemented yet)."
        )


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
