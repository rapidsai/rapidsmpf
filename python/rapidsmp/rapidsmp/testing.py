# Copyright (c) 2025, NVIDIA CORPORATION.
"""Submodule for testing."""

from __future__ import annotations

import io
import os
import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import cudf
import cudf._lib.column
import cudf.testing
import pylibcudf

from rapidsmp.utils.cudf import pylibcudf_to_cudf_dataframe

if TYPE_CHECKING:
    from collections.abc import Generator


def assert_eq(
    left: Any,
    right: Any,
    *,
    ignore_index: bool = True,
    sort_rows: str | None = None,
    **kwargs,
):
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
    kwargs
        Keyword arguments to control behavior of comparisons. See
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
        left = pylibcudf_to_cudf_dataframe(left)
    if isinstance(right, pylibcudf.Table):
        right = pylibcudf_to_cudf_dataframe(right)
    if ignore_index:
        left = left.reset_index(drop=True)
        right = right.reset_index(drop=True)
    if sort_rows is not None:
        left = left.sort_values(by=sort_rows, ignore_index=ignore_index)
        right = right.sort_values(by=sort_rows, ignore_index=ignore_index)
    return cudf.testing.assert_eq(left, right, **kwargs)


@contextmanager
def capture_output(
    fileno: int = sys.stdout.fileno(),
) -> Generator[io.StringIO, None, None]:
    """
    Context manager to capture the output written to a file descriptor.

    This context manager supports output from C/C++ extensions unlike pytest's capsys.

    Parameters
    ----------
    fileno
        The file descriptor to capture output from.

    Yields
    ------
    io.StringIO
        An in-memory string buffer containing the captured output.
    """
    saved_stdout_fd = os.dup(fileno)
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, fileno)
    os.close(write_fd)

    output = io.StringIO()
    try:
        yield output
    finally:
        os.dup2(saved_stdout_fd, fileno)
        os.close(saved_stdout_fd)

    with os.fdopen(read_fd) as f:
        output.write(f.read())
    output.seek(0)
