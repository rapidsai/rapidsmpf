# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Shuffler integration with external libraries."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, TypeVar, runtime_checkable

from rapidsmpf.buffer.spill_collection import SpillCollection
from rapidsmpf.config import Options

if TYPE_CHECKING:
    from rapidsmpf.buffer.resource import BufferResource
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.progress_thread import ProgressThread
    from rapidsmpf.shuffler import Shuffler
    from rapidsmpf.statistics import Statistics


DataFrameT = TypeVar("DataFrameT")


@dataclass
class WorkerContext:
    """
    RapidsMPF specific attributes for a worker.

    Attributes
    ----------
    lock
        The global worker lock. Must be acquired before accessing attributes
        that might be modified while the worker is running such as the shufflers.
    br
        The buffer resource used by the worker exclusively.
    progress_thread
        The progress thread used by the worker.
    comm
        The communicator connected to all other workers.
    spill_collection
        A collection of Python objects that can be spilled to free up device memory.
    statistics
        The statistics used by the worker. If None, statistics is disabled.
    shufflers
        A mapping from shuffler IDs to active shuffler instances.
    options
        Configuration options.
    """

    lock: ClassVar[threading.RLock] = threading.RLock()
    br: BufferResource | None = None
    progress_thread: ProgressThread | None = None
    comm: Communicator | None = None
    spill_collection: SpillCollection = field(default_factory=SpillCollection)
    statistics: Statistics | None = None
    shufflers: dict[int, Shuffler] = field(default_factory=dict)
    options: Options = field(default_factory=Options)


@runtime_checkable
class ShuffleIntegration(Protocol[DataFrameT]):
    """
    dask-integration protocol.

    This protocol can be used to implement a RapidsMPF-shuffle
    operation using a Dask task graph.
    """

    @staticmethod
    def insert_partition(
        df: DataFrameT,
        partition_id: int,
        partition_count: int,
        shuffler: Shuffler,
        options: Any,
        *other: Any,
    ) -> None:
        """
        Add a partition to a RapidsMPF Shuffler.

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

    @staticmethod
    def extract_partition(
        partition_id: int,
        shuffler: Shuffler,
        options: Any,
    ) -> DataFrameT:
        """
        Extract a DataFrame partition from a RapidsMPF Shuffler.

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
