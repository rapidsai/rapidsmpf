# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""cudf utilities."""

from __future__ import annotations

import cudf
import cudf.core.column
import cudf.testing
import pylibcudf


def cudf_to_pylibcudf_table(df: cudf.DataFrame) -> pylibcudf.Table:
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


def pylibcudf_to_cudf_dataframe(
    table: pylibcudf.Table,
    column_names: list[str] | None = None,
) -> cudf.DataFrame:
    """
    Convert a pylibcudf Table to a cuDF DataFrame.

    Parameters
    ----------
    table
        The pylibcudf Table to convert.
    column_names
        List of column names.

    Returns
    -------
    cudf.DataFrame
        A cuDF DataFrame representation of the input Table.
    """
    data = {
        str(i): cudf.core.column.ColumnBase.from_pylibcudf(col)
        for i, col in enumerate(table.columns())
    }
    return cudf.DataFrame._from_data(data, columns=column_names)
