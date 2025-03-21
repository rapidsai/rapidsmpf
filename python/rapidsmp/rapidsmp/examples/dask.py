# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Dask-cuDF integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import dask.dataframe as dd
from dask.tokenize import tokenize

import rmm.mr
from rmm.pylibrmm.stream import DEFAULT_STREAM

from rapidsmp.integrations.dask import rapidsmp_shuffle_graph
from rapidsmp.shuffler import partition_and_pack, unpack_and_concat
from rapidsmp.testing import pylibcudf_to_cudf_dataframe

if TYPE_CHECKING:
    from collections.abc import Sequence

    import dask_cudf

    import cudf

    from rapidsmp.shuffler import Shuffler


class DaskCudfIntegration:
    """
    Dask-cuDF protocol for Dask integration.

    This protocol can be used to implement a rapidsmp-shuffle
    operation on a Dask-cuDF collection.

    See Also
    --------
    rapidsmp.integrations.dask.DaskIntegration
        Base Dask-integration protocol definition.
    """

    @staticmethod
    def insert_partition(
        df: cudf.DataFrame,
        on: Sequence[str],
        partition_count: int,
        shuffler: Shuffler,
    ) -> None:
        """
        Add cudf DataFrame chunks to an RMP shuffler.

        Parameters
        ----------
        df
            DataFrame partition to add to a rapidsmp shuffler.
        on
            Sequence of column names to shuffle on.
        partition_count
            Number of output partitions for the current shuffle.
        shuffler
            The rapidsmp Shuffler object to extract from.
        """
        columns_to_hash = tuple(list(df.columns).index(val) for val in on)
        packed_inputs = partition_and_pack(
            df.to_pylibcudf()[0],
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
    ) -> cudf.DataFrame:
        """
        Extract a finished partition from the RMP shuffler.

        Parameters
        ----------
        partition_id
            Partition id to extract.
        column_names
            Sequence of output column names.
        shuffler
            The rapidsmp Shuffler object to extract from.

        Returns
        -------
        A shuffled DataFrame partition.
        """
        shuffler.wait_on(partition_id)
        table = unpack_and_concat(
            shuffler.extract(partition_id),
            stream=DEFAULT_STREAM,
            device_mr=rmm.mr.get_current_device_resource(),
        )
        return pylibcudf_to_cudf_dataframe(
            table,
            column_names=column_names,
        )


def dask_cudf_shuffle(
    df: dask_cudf.DataFrame,
    shuffle_on: list[str],
    *,
    partition_count: int | None = None,
) -> dask_cudf.DataFrame:
    """
    Shuffle a dask_cudf.DataFrame with rapidsmp.

    Parameters
    ----------
    df
        Input `dask_cudf.DataFrame` collection.
    shuffle_on
        List of column names to shuffle on.
    partition_count
        Output partition count. Default will preserve
        the input partition count.

    Returns
    -------
    Shuffled Dask-cuDF DataFrame collection.
    """
    df0 = df.optimize()
    count_in = df0.npartitions
    count_out = partition_count or count_in
    token = tokenize(df0, shuffle_on, count_out)
    name_in = df0._name
    name_out = f"shuffle-{token}"
    graph = rapidsmp_shuffle_graph(
        name_in,
        name_out,
        list(df0.columns),
        shuffle_on,
        count_in,
        count_out,
        DaskCudfIntegration,
    )

    # Add df0 dependencies to the task graph
    graph.update(df0.dask)

    # Return a Dask-DataFrame collection
    return dd.from_graph(
        graph,
        df0._meta,
        (None,) * (count_out + 1),
        [(name_out, pid) for pid in range(count_out)],
        "rapidsmp",
    )
