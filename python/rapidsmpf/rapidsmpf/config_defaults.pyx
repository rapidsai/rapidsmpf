# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
String-form default values for config options.

Defaults are stored as strings and parsed through the same factories used
for user-supplied values.

Examples
--------
>>> from rapidsmpf.config_defaults import DEFAULTS
>>> DEFAULTS["num_streams"]
'16'
>>> DEFAULTS["statistics"]
'false'
"""

from types import MappingProxyType

from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map


cdef extern from "<rapidsmpf/config_defaults.hpp>" nogil:
    const unordered_map[string, string] _CPP_DEFAULTS \
        "rapidsmpf::config::DEFAULTS"


cdef _build_defaults():
    cdef unordered_map[string, string] m = _CPP_DEFAULTS
    return MappingProxyType(
        {k.decode("utf-8"): v.decode("utf-8") for k, v in m}
    )


DEFAULTS = _build_defaults()


__all__ = ["DEFAULTS"]
