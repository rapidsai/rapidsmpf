# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Submodule for testing."""

from __future__ import annotations

import pylibcudf


def assert_eq_with_plc(
    left: pylibcudf.Table,
    right: pylibcudf.Table,
    *,
    sort_rows: int | None = None,
) -> None:
    """
    Assert that two tables are equivalent using pylibcudf.

    Parameters
    ----------
    left
        plc.Table to compare.
    right
        plc.Table to compare.
    sort_rows
        If not None, sort both tables by this column before comparing.
        An ``int`` is treated as a column index.

    Raises
    ------
    AssertionError
        If the two tables do not compare equal.
    """
    if sort_rows is not None:
        column_order = [pylibcudf.types.Order.ASCENDING]
        null_precedence = [pylibcudf.types.NullOrder.BEFORE]
        left = pylibcudf.sorting.stable_sort_by_key(
            left,
            pylibcudf.Table([left.columns()[sort_rows]]),
            column_order,
            null_precedence,
        )
        right = pylibcudf.sorting.stable_sort_by_key(
            right,
            pylibcudf.Table([right.columns()[sort_rows]]),
            column_order,
            null_precedence,
        )
    if not pylibcudf.table_equality.tables_equal(left, right):
        raise AssertionError(f"Table are not equal with {sort_rows=}")
