# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Submodule for testing."""

from __future__ import annotations

from typing import Any

import cudf
import cudf.testing
import pylibcudf

from rapidsmpf.utils.cudf import pylibcudf_to_cudf_dataframe


def assert_eq(
    left: Any,
    right: Any,
    *,
    ignore_index: bool = True,
    sort_rows: str | None = None,
    **kwargs: dict[str, Any],
) -> None:
    """
    Assert that two cudf/pylibcudf-like things are equivalent.

    Parameters
    ----------
    left : dataframe-like
        Object to compare.
    right : dataframe-like
        Object to compare.
    ignore_index
        Ignore the index when comparing.
    sort_rows
        If not None, sort the rows by the specified column before comparing.
    **kwargs
        Keyword arguments to control behavior of comparisons. See
        :func:`assert_frame_equal`, :func:`assert_series_equal`, and
        :func:`assert_index_equal`.

    Raises
    ------
    AssertionError
        If the two objects do not compare equal.

    Notes
    -----
    This equality test works for pandas/cudf dataframes/series/indexes/scalars
    in the same way, and so makes it easier to perform parametrized testing
    without switching between assert_frame_equal/assert_series_equal/...
    functions.
    """
    if isinstance(left, pylibcudf.Table):
        left = pylibcudf_to_cudf_dataframe(left)
    if isinstance(right, pylibcudf.Table):
        right = pylibcudf_to_cudf_dataframe(right)
    if ignore_index:
        left = left.reset_index(drop=True)
        right = right.reset_index(drop=True)
    if sort_rows is not None:
        left = left.sort_values(by=sort_rows, ignore_index=ignore_index)
        right = right.sort_values(by=sort_rows, ignore_index=ignore_index)
    cudf.testing.assert_eq(left, right, **kwargs)
