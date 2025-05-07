# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Dask-cuDF integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import dask.dataframe as dd
import numpy as np
from dask.tokenize import tokenize
from dask.utils import M

import rmm.mr
from rmm.pylibrmm.stream import DEFAULT_STREAM

from rapidsmpf.integrations.dask.shuffler import rapidsmpf_shuffle_graph
from rapidsmpf.shuffler import partition_and_pack, split_and_pack, unpack_and_concat
from rapidsmpf.testing import pylibcudf_to_cudf_dataframe

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    import dask_cudf

    import cudf

    from rapidsmpf.shuffler import Shuffler


class DaskCudfIntegration:
    """
    Dask-cuDF protocol for Dask integration.

    This protocol can be used to implement a RapidsMPF-shuffle
    operation on a Dask-cuDF collection.

    See Also
    --------
    rapidsmpf.integrations.dask.shuffler.DaskIntegration
        Base Dask-integration protocol definition.
    """

    @staticmethod
    def insert_partition(
        df: cudf.DataFrame,
        on: Sequence[str],
        partition_count: int,
        shuffler: Shuffler,
        sort_boundaries: cudf.DataFrame | None,
        options: dict[str, Any] | None,
    ) -> None:
        """
        Add cudf DataFrame chunks to an RMPF shuffler.

        Parameters
        ----------
        df
            DataFrame partition to add to a RapidsMPF shuffler.
        on
            Sequence of column names to shuffle on.
        partition_count
            Number of output partitions for the current shuffle.
        shuffler
            The RapidsMPF Shuffler object to extract from.
        sort_boundaries
            Output partition boundaries for sorting.
        options
            Optional key-work arguments.
        """
        if sort_boundaries is None:
            if options:
                raise ValueError(f"Unsupported options: {options}")
            columns_to_hash = tuple(list(df.columns).index(val) for val in on)
            packed_inputs = partition_and_pack(
                df.to_pylibcudf()[0],
                columns_to_hash=columns_to_hash,
                num_partitions=partition_count,
                stream=DEFAULT_STREAM,
                device_mr=rmm.mr.get_current_device_resource(),
            )
        else:
            # TODO: Check for unsupported kwargs
            df = df.sort_values(on)
            splits = df[on[0]].searchsorted(sort_boundaries, side="right")
            packed_inputs = split_and_pack(
                df.to_pylibcudf()[0],
                splits.tolist(),
                stream=DEFAULT_STREAM,
                device_mr=rmm.mr.get_current_device_resource(),
            )
        shuffler.insert_chunks(packed_inputs)

    @staticmethod
    def extract_partition(
        partition_id: int,
        column_names: list[str],
        shuffler: Shuffler,
        options: dict[str, Any] | None,
    ) -> cudf.DataFrame:
        """
        Extract a finished partition from the RMPF shuffler.

        Parameters
        ----------
        partition_id
            Partition id to extract.
        column_names
            Sequence of output column names.
        shuffler
            The RapidsMPF Shuffler object to extract from.
        options
            Additional options.

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
    sort: bool = False,
    partition_count: int | None = None,
) -> dask_cudf.DataFrame:
    """
    Shuffle a dask_cudf.DataFrame with RapidsMPF.

    Parameters
    ----------
    df
        Input `dask_cudf.DataFrame` collection.
    shuffle_on
        List of column names to shuffle on.
    sort
        Whether the output partitioning should be in
        sorted order. The first column in ``shuffle_on``
        must be numerical.
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
    if sort:
        boundaries = (
            df0[shuffle_on[0]].quantile(np.linspace(0.0, 1.0, count_out)[1:]).optimize()
        )
        sort_boundaries_name = (boundaries._name, 0)
    else:
        sort_boundaries_name = None
    graph = rapidsmpf_shuffle_graph(
        name_in,
        name_out,
        list(df0.columns),
        shuffle_on,
        count_in,
        count_out,
        DaskCudfIntegration,
        sort_boundaries_name=sort_boundaries_name,
    )

    # Add df0 dependencies to the task graph
    graph.update(df0.dask)
    if sort:
        graph.update(boundaries.dask)

    shuffled = dd.from_graph(
        graph,
        df0._meta,
        (None,) * (count_out + 1),
        [(name_out, pid) for pid in range(count_out)],
        "rapidsmpf",
    )

    # Return a Dask-DataFrame collection
    if sort:
        return shuffled.map_partitions(
            M.sort_values(shuffle_on),
            meta=shuffled._meta,
        )
    else:
        return shuffled
