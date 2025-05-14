# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.utility cimport move

import os
import re


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
