# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Submodule for testing."""

from __future__ import annotations

from typing import Any

import pyarrow as pa

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


def assert_eq_with_pyarrow(
    left: pa.Table | pylibcudf.Table,
    right: pa.Table | pylibcudf.Table,
    *,
    sort_rows: int | str | None = None,
) -> None:
    """
    Assert that two tables are equivalent using pyarrow.

    Each input may be either a :class:`pyarrow.Table` or a
    :class:`pylibcudf.Table`. ``pylibcudf.Table`` inputs are converted to
    :class:`pyarrow.Table` (no column metadata is attached, so converted
    tables have empty column names).

    Parameters
    ----------
    left
        Object to compare.
    right
        Object to compare.
    sort_rows
        If not None, sort both tables by this column before comparing.
        An ``int`` is treated as a column index; a ``str`` is resolved to
        a column index against the left-hand schema after conversion.

    Raises
    ------
    AssertionError
        If the two tables do not compare equal.
    """
    if isinstance(left, pylibcudf.Table):
        left = left.to_arrow()
    if isinstance(right, pylibcudf.Table):
        right = right.to_arrow()
    if sort_rows is not None:
        idx = (
            sort_rows
            if isinstance(sort_rows, int)
            else left.schema.names.index(sort_rows)
        )
        left = left.take(pa.compute.sort_indices(left.column(idx)))
        right = right.take(pa.compute.sort_indices(right.column(idx)))
    assert left.equals(right), f"{left}\n!=\n{right}"
