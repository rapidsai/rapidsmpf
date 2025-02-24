# Copyright (c) 2025, NVIDIA CORPORATION.
"""Dask + cudf-Polars integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cudf_polars.containers import DataFrame

import rmm.mr
from rmm.pylibrmm.stream import DEFAULT_STREAM

from rapidsmp.integrations.dask import DaskIntegration
from rapidsmp.shuffler import partition_and_pack, unpack_and_concat

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rapidsmp.shuffler import Shuffler


class CudfPolarsIntegration(DaskIntegration):
    """cuDF Polars protocol for Dask integration."""

    @staticmethod
    def insert_partition_hash(
        df: DataFrame,
        on: Sequence[str],
        partition_count: int,
        shuffler: Shuffler,
    ) -> None:
        """Add cudf-polars DataFrame chunks to an RMP shuffler."""
        columns_to_hash = tuple(df.column_names.index(val) for val in on)
        packed_inputs = partition_and_pack(
            df.table,
            columns_to_hash=columns_to_hash,
            num_partitions=partition_count,
            stream=DEFAULT_STREAM,
            device_mr=rmm.mr.get_current_device_resource(),
        )
        shuffler.insert_chunks(packed_inputs)

    @staticmethod
    def extract_partition(
        partition_id: int,
        column_names: list[str],
        shuffler: Shuffler,
    ) -> DataFrame:
        """Extract a finished partition from the RMP shuffler."""
        shuffler.wait_on(partition_id)
        return DataFrame.from_table(
            unpack_and_concat(
                shuffler.extract(partition_id),
                stream=DEFAULT_STREAM,
                device_mr=rmm.mr.get_current_device_resource(),
            ),
            column_names,
        )
