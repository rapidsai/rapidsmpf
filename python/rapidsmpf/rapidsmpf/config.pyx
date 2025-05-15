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
                    return default_value;
                }
                return rapidsmpf::parse_string<T>(x);
            }
        );
    }
    """
    T cpp_options_get_using_parse_string[T](
        cpp_Options options, string key, T default_value
    ) nogil


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

    def get_or_default(self, str key, default_value):
        """
        Get the value associated with the given key, or the default value.

        The type of the returned value is determined by the type of `default_value`.

        Parameters
        ----------
        key
            The key to look up in the configuration.
        default_value
            The value to return if the key is not found.

        Returns
        -------
        The value associated with the key, or the default value if the key does
        not exist.

        Raises
        ------
        ValueError
            If the type of default_value isn't supported.
        """
        if isinstance(default_value, bool):
            return cpp_options_get_using_parse_string[bool_t](
                self._handle, str.encode(key), <bool_t?>default_value
            )
        elif isinstance(default_value, float):
            return cpp_options_get_using_parse_string[double](
                self._handle, str.encode(key), <double?>default_value
            )
        elif isinstance(default_value, int):
            return cpp_options_get_using_parse_string[int64_t](
                self._handle, str.encode(key), <int64_t?>default_value
            )
        elif isinstance(default_value, str):
            return cpp_options_get_using_parse_string[string](
                self._handle, str.encode(key), str.encode(default_value)
            ).decode('UTF-8')
        raise ValueError(
            f"default type ({type(default_value)}) is not support, "
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
