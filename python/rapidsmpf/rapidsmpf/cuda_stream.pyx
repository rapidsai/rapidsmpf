# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.stream cimport Stream


def is_equal_streams(Stream s1 not None, Stream s2 not None):
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
    return s1.view().value() == s2.view().value()
