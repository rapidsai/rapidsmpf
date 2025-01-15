# Copyright (c) 2025, NVIDIA CORPORATION.
"""Submodule for testing."""

from __future__ import annotations

import cudf
import cudf._lib.column
import cudf.testing
import pylibcudf


def to_pylibcudf_table(df: cudf.DataFrame) -> pylibcudf.Table:
    """
    Convert a cuDF DataFrame to a pylibcudf Table (read-only).

    Parameters
    ----------
    df
        The cuDF DataFrame to convert.

    Returns
    -------
    pylibcudf.Table
        A pylibcudf Table representation of the input DataFrame.
    """
    return pylibcudf.Table([col.to_pylibcudf(mode="read") for col in df._columns])


def to_cudf_dataframe(table: pylibcudf.Table) -> cudf.DataFrame:
    """
    Convert a pylibcudf Table to a cuDF DataFrame.

    Parameters
    ----------
    table
        The pylibcudf Table to convert.

    Returns
    -------
    cudf.DataFrame
        A cuDF DataFrame representation of the input Table.
    """
    data = {
        str(i): cudf._lib.column.Column.from_pylibcudf(col)
        for i, col in enumerate(table.columns())
    }
    return cudf.DataFrame._from_data(data)


def assert_eq(left, right, **kwargs):
    """
    Assert that two cudf/pylibcudf-like things are equivalent.

    Parameters
    ----------
    left
        Object to compare
    right
        Object to compare
    kwargs
        Keyword arguments to control behaviour of comparisons. See
        :func:`assert_frame_equal`, :func:`assert_series_equal`, and
        :func:`assert_index_equal`.

    Notes
    -----
    This equality test works for pandas/cudf dataframes/series/indexes/scalars
    in the same way, and so makes it easier to perform parametrized testing
    without switching between assert_frame_equal/assert_series_equal/...
    functions.

    Raises
    ------
    AssertionError
        If the two objects do not compare equal.
    """
    if isinstance(left, pylibcudf.Table):
        left = to_cudf_dataframe(left)
    if isinstance(right, pylibcudf.Table):
        right = to_cudf_dataframe(right)
    return cudf.testing.assert_eq(left, right, **kwargs)
