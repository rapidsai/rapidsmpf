# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.stream cimport Stream


cpdef bint is_equal_streams(Stream s1, Stream s2):
    """
    Check whether two streams refer to the same underlying CUDA stream.

    Parameters
    ----------
    s1
        First stream to compare.
    s2
        Second stream to compare.

    Returns
    -------
    ``True`` if both inputs reference the same stream; ``False`` otherwise.
    """
    if s1 is None:
        raise ValueError("s1 cannot be None")
    if s2 is None:
        raise ValueError("s2 cannot be None")
    return s1.view().value() == s2.view().value()
