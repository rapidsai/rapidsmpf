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

from rapidsmpf.integrations.cudf.partition import (
    partition_and_pack,
    split_and_pack,
    unpack_and_concat,
)
from rapidsmpf.integrations.dask.shuffler import rapidsmpf_shuffle_graph
from rapidsmpf.testing import pylibcudf_to_cudf_dataframe

if TYPE_CHECKING:
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
        partition_id: int,
        partition_count: int,
        shuffler: Shuffler,
        options: dict[str, Any],
        *other: Any,
    ) -> None:
        """
        Add cudf DataFrame chunks to an RMPF shuffler.

        Parameters
        ----------
        df
            DataFrame partition to add to a RapidsMPF shuffler.
        partition_id
            The input partition id of ``df``.
        partition_count
            Number of output partitions for the current shuffle.
        shuffler
            The RapidsMPF Shuffler object to extract from.
        options
            Additional options.
        *other
            Other data needed for partitioning. For example,
            this may be boundary values needed for sorting.
        """
        on = options["on"]
        if other:
            df = df.sort_values(on)
            (sort_boundaries,) = other
            splits = df[on[0]].searchsorted(sort_boundaries, side="right")
            packed_inputs = split_and_pack(
                df.to_pylibcudf()[0],
                splits.tolist(),
                stream=DEFAULT_STREAM,
                device_mr=rmm.mr.get_current_device_resource(),
            )
        else:
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
        shuffler: Shuffler,
        options: dict[str, Any],
    ) -> cudf.DataFrame:
        """
        Extract a finished partition from the RMPF shuffler.

        Parameters
        ----------
        partition_id
            Partition id to extract.
        shuffler
            The RapidsMPF Shuffler object to extract from.
        options
            Additional options.

        Returns
        -------
        A shuffled DataFrame partition.
        """
        column_names = options["column_names"]
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
    on: list[str],
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
    on
        List of column names to shuffle on.
    sort
        Whether the output partitioning should be in
        sorted order. The first column in ``on`` must
        be numerical when ``sort`` is True.
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
    token = tokenize(df0, on, count_out)
    name_in = df0._name
    name_out = f"shuffle-{token}"
    sort_boundary_names: tuple[()] | tuple[tuple[Any, int]]
    if sort:
        # NOTE: This implementation makes no effort to
        # handle the case where many rows are equal
        # (they will all be mapped to the same partition).
        boundaries = (
            df0[on[0]].quantile(np.linspace(0.0, 1.0, count_out)[1:]).optimize()
        )
        sort_boundary_names = ((boundaries._name, 0),)
    else:
        sort_boundary_names = ()
    graph = rapidsmpf_shuffle_graph(
        name_in,
        name_out,
        count_in,
        count_out,
        DaskCudfIntegration,
        {"on": on, "column_names": list(df0.columns)},
        *sort_boundary_names,
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
            M.sort_values,
            on,
            meta=shuffled._meta,
        )
    else:
        return shuffled
