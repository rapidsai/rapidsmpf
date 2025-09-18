# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Dask-cuDF integration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import dask.dataframe as dd
import numpy as np
from dask.tokenize import tokenize
from dask.utils import M

from rmm.pylibrmm.stream import DEFAULT_STREAM

import rapidsmpf.integrations.dask
import rapidsmpf.integrations.single
from rapidsmpf.config import Options
from rapidsmpf.integrations.core import get_shuffler
from rapidsmpf.integrations.cudf.partition import (
    partition_and_pack,
    split_and_pack,
    unpack_and_concat,
    unspill_partitions,
)
from rapidsmpf.testing import pylibcudf_to_cudf_dataframe
from rapidsmpf.utils.cudf import cudf_to_pylibcudf_table

if TYPE_CHECKING:
    from typing import Any

    import dask_cudf

    import cudf

    from rapidsmpf.integrations.core import ShufflerIntegration, WorkerContext
    from rapidsmpf.shuffler import Shuffler


class DaskCudfIntegration:
    """
    Dask-cuDF protocol for Dask integration.

    This protocol can be used to implement a RapidsMPF-shuffle
    operation on a Dask-cuDF collection.

    See Also
    --------
    rapidsmpf.integrations.core.ShufflerIntegration
        Base shuffler-integration protocol definition.
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
        if options.get("cluster_kind", "distributed") == "distributed":
            ctx = rapidsmpf.integrations.dask.get_worker_context()
        else:
            ctx = rapidsmpf.integrations.single.get_worker_context()

        assert ctx.br is not None
        on = options["on"]
        if other:
            df = df.sort_values(on)
            (sort_boundaries,) = other
            splits = df[on[0]].searchsorted(sort_boundaries, side="right")
            packed_inputs = split_and_pack(
                cudf_to_pylibcudf_table(df),
                splits.tolist(),
                br=ctx.br,
                stream=DEFAULT_STREAM,
            )
        else:
            columns_to_hash = tuple(list(df.columns).index(val) for val in on)
            packed_inputs = partition_and_pack(
                cudf_to_pylibcudf_table(df),
                columns_to_hash=columns_to_hash,
                num_partitions=partition_count,
                br=ctx.br,
                stream=DEFAULT_STREAM,
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
        get_worker_context
            A callable that runs on the worker to get its current
            context.

        Returns
        -------
        A shuffled DataFrame partition.
        """
        if options.get("cluster_kind", "distributed") == "distributed":
            ctx = rapidsmpf.integrations.dask.get_worker_context()
        else:
            ctx = rapidsmpf.integrations.single.get_worker_context()

        assert ctx.br is not None
        column_names = options["column_names"]
        shuffler.wait_on(partition_id)
        table = unpack_and_concat(
            unspill_partitions(
                shuffler.extract(partition_id),
                stream=DEFAULT_STREAM,
                br=ctx.br,
                allow_overbooking=True,
                statistics=ctx.statistics,
            ),
            br=ctx.br,
            stream=DEFAULT_STREAM,
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
    cluster_kind: Literal["distributed", "single", "auto"] = "auto",
    config_options: Options = Options(),
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
    cluster_kind
        What kind of Dask cluster to shuffle on. Available
        options are ``{'distributed', 'single', 'auto'}``.
        If 'auto' (the default), 'distributed' will be
        used if a global Dask client is found.
    config_options
        RapidsMPF configuration options.

    Returns
    -------
    Shuffled Dask-cuDF DataFrame collection.

    Notes
    -----
    This API is currently intended for demonstration and
    testing purposes only.
    """
    if cluster_kind not in ("distributed", "single", "auto"):
        raise ValueError(
            f"Expected one of 'distributed', 'single', or 'auto'. Got {cluster_kind}"
        )

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

    if cluster_kind == "auto":
        try:
            from distributed import get_client

            get_client()
        except (ImportError, ValueError):
            # Failed to import distributed/dask-cuda or find a Dask client.
            # Use single shuffle instead.
            cluster_kind = "single"
        else:
            cluster_kind = "distributed"

    if cluster_kind == "distributed":
        shuffle = rapidsmpf.integrations.dask.rapidsmpf_shuffle_graph
    else:
        shuffle = rapidsmpf.integrations.single.rapidsmpf_shuffle_graph

    shuffle_graph_args = (
        name_in,
        name_out,
        count_in,
        count_out,
        DaskCudfIntegration,
        {"on": on, "column_names": list(df0.columns), "cluster_kind": cluster_kind},
        *sort_boundary_names,
    )

    graph = shuffle(*shuffle_graph_args, config_options=config_options)

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


class DaskCudfJoinIntegration:
    """Dask-cuDF protocol for unified join integration."""

    @staticmethod
    def shuffler_integration() -> ShufflerIntegration[cudf.DataFrame]:
        """Return the shuffler integration."""
        return DaskCudfIntegration()

    @classmethod
    def join_partition(
        cls,
        ctx: WorkerContext,
        bcast_side: Literal["left", "right", "none"],
        left_op_id: int,
        right_op_id: int,
        part_id: int,
        n_worker_tasks: int,
        options: Any,
    ) -> cudf.DataFrame:
        """
        Produce a joined cudf.DataFrame partition.

        Parameters
        ----------
        ctx
            The worker context.
        bcast_side
            The side of the join being broadcasted (if either).
        left_op_id
            The ID of the left shuffle.
        right_op_id
            The ID of the right shuffle.
        part_id
            The ID of the partition being joined.
        n_worker_tasks
            The number of join_partition tasks to be called on this worker.
            This information may be used for cleanup.
        options
            Additional options.

        Returns
        -------
        A DataFrame containing the joined data.

        Notes
        -----
        This method is used to produce a single joined table chunk.
        """
        assert ctx.br is not None
        if bcast_side != "none":
            raise NotImplementedError("Broadcast join not implemented.")

        # Extract left side
        try:
            left_shuffler = get_shuffler(ctx, left_op_id)
            left = cls.shuffler_integration().extract_partition(
                part_id,
                left_shuffler,
                {"column_names": options["left_column_names"]},
            )
        finally:
            if left_shuffler.finished():
                with ctx.lock:
                    if left_op_id in ctx.shufflers:
                        del ctx.shufflers[left_op_id]

        # Extract right side
        try:
            right_shuffler = get_shuffler(ctx, right_op_id)
            right = cls.shuffler_integration().extract_partition(
                part_id,
                right_shuffler,
                {"column_names": options["right_column_names"]},
            )
        finally:
            if right_shuffler.finished():
                with ctx.lock:
                    if right_op_id in ctx.shufflers:
                        del ctx.shufflers[right_op_id]

        # Return merged result
        kwargs = {
            "left_on": options["left_on"],
            "right_on": options["right_on"],
            "how": options["how"],
        }
        return left.merge(right, **kwargs)


def dask_cudf_join(
    left: dask_cudf.DataFrame,
    right: dask_cudf.DataFrame,
    left_on: list[str],
    right_on: list[str],
    bcast_side: Literal["left", "right", "none"] = "none",
    *,
    how: Literal["inner", "left", "right"] = "inner",
    config_options: Options = Options(),
) -> dask_cudf.DataFrame:
    """Join two Dask-cuDF DataFrames with RapidsMPF."""
    from rapidsmpf.integrations.dask.join import rapidsmpf_join_graph

    if bcast_side != "none":
        raise ValueError("Only 'none' is supported for now.")

    left0 = left.optimize()
    right0 = right.optimize()
    left_count_in = left0.npartitions
    right_count_in = right0.npartitions
    count_out = max(left_count_in, right_count_in)  # TODO: May be different for bcast

    token = tokenize(left0, right0, left_on, bcast_side, right_on, how)
    left_name_in = left0._name
    right_name_in = right0._name
    name_out = f"unified-join-{token}"
    graph = rapidsmpf_join_graph(
        left_name_in,
        right_name_in,
        name_out,
        bcast_side,
        left_count_in,
        right_count_in,
        DaskCudfJoinIntegration(),
        {
            "left_column_names": left0.columns,
            "right_column_names": right0.columns,
            "left_on": left_on,
            "right_on": right_on,
            "how": how,
        },
        config_options=config_options,
    )
    graph.update(left0.dask)
    graph.update(right0.dask)

    meta = left0.merge(right0, left_on=left_on, right_on=right_on, how=how)._meta
    return dd.from_graph(
        graph,
        meta,
        (None,) * (count_out + 1),
        [(name_out, pid) for pid in range(count_out)],
        "rapidsmpf",
    )
