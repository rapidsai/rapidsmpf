# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Example running a Bulk RapidsMPF Shuffle operation using Ray and UCXX communication."""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pylibcudf as plc
import ray
from cudf_streaming.partition_utils import (
    partition_and_pack,
    unpack_and_concat,
)

import rmm.mr

from rapidsmpf.config import Options
from rapidsmpf.integrations.ray import RapidsMPFActor, setup_ray_ucxx_cluster
from rapidsmpf.memory.buffer_resource import BufferResource
from rapidsmpf.memory.spill import unspill_partitions
from rapidsmpf.shuffler import Shuffler
from rapidsmpf.statistics import Statistics
from rapidsmpf.utils.string import format_bytes, parse_bytes

if TYPE_CHECKING:
    from collections.abc import Iterator


@ray.remote(num_gpus=1, num_cpus=4)
class BulkRayShufflerActor(RapidsMPFActor):
    """
    Actor that performs a bulk shuffle operation using Ray.

    Parameters
    ----------
    nranks
        Number of ranks in the communication group.
    total_nparts
        Total number of output partitions.
    shuffle_on
        List of column names to shuffle on.
    batchsize
        Number of files to process in a batch.
    output_path
        Path to write output files.
    rmm_pool_size
        Size of the RMM memory pool in bytes.
    spill_device
        Device memory limit for spilling to host in bytes.
    rmm_async
        Whether to use RMM's cudaMallocAsync-backed memory resource.
    pinned_memory
        Whether to use pinned host memory for spilling when available.
    pinned_initial_pool_size
        Initial pinned host memory pool size in bytes.
    enable_statistics
        Whether to collect statistics.
    """

    def __init__(
        self,
        nranks: int,
        total_nparts: int,
        shuffle_on: list[str],
        batchsize: int = 1,
        output_path: str = "./",
        rmm_pool_size: int = 1024 * 1024 * 1024,
        spill_device: int | None = None,
        *,
        rmm_async: bool = False,
        pinned_memory: bool = False,
        pinned_initial_pool_size: int | None = None,
        enable_statistics: bool = False,
    ):
        self.batchsize = batchsize
        self.shuffle_on = shuffle_on
        self.output_path = output_path
        self.total_nparts = total_nparts
        self.rmm_pool_size = rmm_pool_size
        self.rmm_async = rmm_async
        self.spill_device = spill_device
        self.pinned_memory = pinned_memory
        self.pinned_initial_pool_size = pinned_initial_pool_size

        # Build Options for BufferResource so it can handle pinned memory and
        # spill device limit internally — spilling works with or without pinned memory.
        br_options: dict[str, str] = {}
        if self.spill_device is not None:
            br_options["spill_device_limit"] = str(self.spill_device)
        if self.pinned_memory:
            br_options["pinned_memory"] = "True"
        if self.pinned_initial_pool_size is not None:
            br_options["pinned_initial_pool_size"] = str(self.pinned_initial_pool_size)

        base_mr = _make_device_memory_resource(
            self.rmm_pool_size,
            rmm_async=self.rmm_async,
        )
        statistics = Statistics(enable=enable_statistics)
        br = BufferResource.from_options(base_mr, Options(br_options), statistics)
        self.mr = br.device_mr_adaptor()
        rmm.mr.set_current_device_resource(self.mr)
        self.pinned_mr = br.pinned_mr
        self.br = br
        super().__init__(nranks, statistics)

    def setup_worker(self, root_address_bytes: bytes) -> None:
        """
        Setup the UCXX communication and a shuffle operation.

        Parameters
        ----------
        root_address_bytes
            Address of the root worker for UCXX initialization.
        """
        super().setup_worker(root_address_bytes)
        self.shuffler: Shuffler = Shuffler(
            self.comm,
            0,
            total_num_partitions=self.total_nparts,
            br=self.br,
        )

    def cleanup(self) -> None:
        """Cleanup the UCXX communication and the shuffle operation."""
        if self.statistics.enabled:
            report = self.statistics.report(mr=self.mr, pinned_mr=self.pinned_mr)
            print(f"=== Statistics report (rank {self.comm.rank}) ===")
            print(report, flush=True)
            self.comm.logger.info(report)
        if self.shuffler is not None:
            self.shuffler.shutdown()

    def read_batch(self, paths: list[str]) -> tuple[plc.Table, list[str]]:
        """
        Read a single batch of Parquet files.

        Parameters
        ----------
        paths
            List of file paths to the Parquet files.

        Returns
        -------
            A tuple containing the read in table and the column names.
        """
        options = plc.io.parquet.ParquetReaderOptions.builder(
            plc.io.SourceInfo(paths)
        ).build()
        tbl_w_meta = plc.io.parquet.read_parquet(options)
        return (tbl_w_meta.tbl, tbl_w_meta.column_names(include_children=False))

    def write_table(
        self,
        table: plc.Table,
        output_path: str,
        id: int | str,
        column_names: list[str],
    ) -> None:
        """
        Write a pylibcudf Table to a Parquet file.

        Parameters
        ----------
        table
            The table to write.
        output_path
            The path to write the table to.
        id
            Partition id used for naming the output file.
        column_names
            The column names of the table.
        """
        path = f"{output_path}/part.{id}.parquet"
        meta = plc.io.types.TableInputMetadata(table)
        for col_meta, name in zip(meta.column_metadata, column_names, strict=True):
            col_meta.set_name(name)
        plc.io.parquet.write_parquet(
            plc.io.parquet.ParquetWriterOptions.builder(plc.io.SinkInfo([path]), table)
            .metadata(meta)
            .build()
        )

    def insert_chunk(self, table: plc.Table, column_names: list[str]) -> None:
        """
        Insert a pylibcudf Table into the shuffler.

        Parameters
        ----------
        table
            The table to insert.
        column_names
            The column names of the table.
        """
        from rmm.pylibrmm.stream import DEFAULT_STREAM

        columns_to_hash = tuple(column_names.index(val) for val in self.shuffle_on)
        packed_inputs = partition_and_pack(
            table,
            columns_to_hash=columns_to_hash,
            num_partitions=self.total_nparts,
            br=self.br,
            stream=DEFAULT_STREAM,
        )
        self.shuffler.insert_chunks(packed_inputs)

    def read_and_insert(self, paths: list[str]) -> list[str]:
        """
        Read the list of parquet files every batchsize and insert the partitions into the shuffler.

        Parameters
        ----------
        paths
            List of file paths to the Parquet files.

        Returns
        -------
            The column names of the table.
        """
        for i in range(0, len(paths), self.batchsize):
            tbl, column_names = self.read_batch(paths[i : i + self.batchsize])
            self.insert_chunk(tbl, column_names)
        self.insert_finished()
        return column_names

    def insert_finished(self) -> None:
        """Tell the shuffler that we are done inserting data."""
        self.shuffler.insert_finished()
        self.comm.logger.info("Insert finished")

    def extract(self) -> Iterator[tuple[int, plc.Table]]:
        """
        Extract shuffled partitions.

        Returns
        -------
            An iterator over the shuffled partitions.
        """
        from rmm.pylibrmm.stream import DEFAULT_STREAM

        self.shuffler.wait()
        for partition_id in self.shuffler.local_partitions():
            packed_chunks = self.shuffler.extract(partition_id)
            partition = unpack_and_concat(
                unspill_partitions(
                    packed_chunks,
                    br=self.br,
                    allow_overbooking=True,
                ),
                br=self.br,
                stream=DEFAULT_STREAM,
            )
            yield partition_id, partition

    def extract_and_write(self, column_names: list[str]) -> None:
        """
        Extract and write shuffled partitions.

        Parameters
        ----------
        column_names
            The column names of the table.
        """
        for partition_id, partition in self.extract():
            self.write_table(partition, self.output_path, partition_id, column_names)

    def lsh_read_and_insert(
        self,
        partitions: list[list[str]],
        band_range: tuple[int, int],
        minhashes_per_band: int,
        id_field: str,
        minhash_field: str,
    ) -> list[str]:
        """
        Read minhash parquets per partition, hash bands, melt, and insert.

        `partitions` is a list of file groups (typically ~2 GiB each, produced by
        Curator's FilePartitioningStage). Each partition is read as one cudf.read_parquet
        call so reads stay aligned with the user-chosen blocksize.

        band_range is half-open [start, end). All bands in the range are processed in a
        single shuffle pass -- there is no iteration over bands.
        """
        import cudf

        bucket_field = "_bucket_id"
        column_names = [id_field, bucket_field]
        start, end = band_range
        if start < 0 or start >= end:
            raise ValueError(f"Invalid band range: {band_range}")
        for partition_files in partitions:
            df = cudf.read_parquet(partition_files, columns=[id_field, minhash_field])
            if len(df) == 0:
                continue
            id_df = df[[id_field]]
            for k in range(start, end):
                idx = list(range(k * minhashes_per_band, (k + 1) * minhashes_per_band))
                rep = cudf.Series([idx]).repeat(len(id_df))
                id_df[f"_bucket_{k}"] = f"b{k}_" + df[minhash_field].list.take(
                    rep
                ).hash_values(method="md5")
            value_vars = [f"_bucket_{k}" for k in range(start, end)]
            band_df = id_df.melt(
                id_vars=[id_field], value_name=bucket_field, value_vars=value_vars
            )[column_names]
            self.insert_chunk(band_df.to_pylibcudf()[0], column_names)
            del df, id_df, band_df
        self.insert_finished()
        return column_names

    def lsh_extract_and_write(self, id_field: str) -> None:
        """Extract shuffled bucket rows, drop singletons, group ids per bucket, write parquet."""
        import cudf

        bucket_field = "_bucket_id"
        column_names = [id_field, bucket_field]
        for partition_id, partition in self.extract():
            df = cudf.DataFrame.from_pylibcudf(
                partition, metadata={"columns": column_names}
            )
            if len(df) == 0:
                continue
            df = df[df[bucket_field].duplicated(keep=False)]
            if len(df) == 0:
                continue
            grouped = (
                df.groupby(bucket_field)[id_field]
                .agg(list)
                .list.sort_values()
                .reset_index()
            )
            grouped.to_parquet(
                f"{self.output_path}/part.{partition_id}.parquet", index=False
            )
            del df, grouped


def _make_device_memory_resource(
    rmm_pool_size: int,
    *,
    rmm_async: bool,
) -> rmm.mr.DeviceMemoryResource:
    if rmm_async:
        # UCXX may transport same-node device buffers through CUDA IPC.
        return rmm.mr.CudaAsyncMemoryResource(
            initial_pool_size=rmm_pool_size,
            release_threshold=rmm_pool_size,
            enable_ipc=True,
        )

    return rmm.mr.PoolMemoryResource(
        rmm.mr.CudaMemoryResource(),
        initial_pool_size=rmm_pool_size,
        maximum_pool_size=rmm_pool_size,
    )


def bulk_ray_shuffle(
    paths: list[str],
    shuffle_on: list[str],
    output_path: str,
    num_workers: int = 2,
    batchsize: int = 1,
    num_output_files: int | None = None,
    rmm_pool_size: int = 1024 * 1024 * 1024,
    spill_device: int | None = None,
    *,
    rmm_async: bool = False,
    pinned_memory: bool = False,
    pinned_initial_pool_size: int | None = None,
    enable_statistics: bool = False,
) -> None:
    """
    Perform a bulk shuffle operation using Ray and UCXX communication.

    Parameters
    ----------
    paths
        The list of paths to the input files.
    shuffle_on
        The list of column names to shuffle on.
    output_path
        The directory to write the shuffled data.
    num_workers
        The number of workers to use.
    batchsize
        The number of files to read on each rank at once.
    num_output_files
        The number of output files to write.
    rmm_pool_size
        The size of the RMM pool. When rmm_async is enabled, this is used as the
        async resource initial pool size and release threshold.
    spill_device
        Device memory limit for spilling to host.
    rmm_async
        Whether to use RMM's cudaMallocAsync-backed memory resource.
    pinned_memory
        Whether to use pinned host memory for spilling when available.
    pinned_initial_pool_size
        Initial pinned host memory pool size in bytes.
    enable_statistics
        Whether to collect statistics.
    """
    # Initialize the UCXX cluster
    num_input_files = len(paths)
    num_output_files = num_output_files or num_input_files
    total_num_partitions = num_output_files
    files_per_rank = math.ceil(num_input_files / num_workers)

    actors = setup_ray_ucxx_cluster(
        BulkRayShufflerActor,
        num_workers=num_workers,
        total_nparts=total_num_partitions,
        shuffle_on=shuffle_on,
        batchsize=batchsize,
        output_path=output_path,
        enable_statistics=enable_statistics,
        rmm_pool_size=rmm_pool_size,
        rmm_async=rmm_async,
        pinned_memory=pinned_memory,
        pinned_initial_pool_size=pinned_initial_pool_size,
        spill_device=spill_device,
    )
    start_time = time.time()
    insert_tasks = []
    for i, actor in enumerate(actors):
        # Calculate the start and end indices for this actor's files
        start = i * files_per_rank
        # Use min to ensure we don't go beyond the end of the paths list
        end = min(start + files_per_rank, num_input_files)
        insert_tasks.append(actor.read_and_insert.remote(paths[start:end]))
    column_names = ray.get(insert_tasks)
    ray.get(
        [
            actor.extract_and_write.remote(column_name)
            for actor, column_name in zip(actors, column_names, strict=False)
        ]
    )
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    ray.get([actor.cleanup.remote() for actor in actors])


def lsh_bulk_ray_shuffle(
    partitions: list[list[str]],
    output_path: str,
    num_bands: int,
    minhashes_per_band: int,
    id_field: str,
    minhash_field: str,
    num_workers: int = 8,
    num_output_files: int | None = None,
    rmm_pool_size: int = 1024 * 1024 * 1024,
    spill_device: int | None = None,
    *,
    rmm_async: bool = False,
    pinned_memory: bool = False,
    pinned_initial_pool_size: int | None = None,
    enable_statistics: bool = False,
) -> None:
    """
    LSH-mode driver: read minhashes per partition, hash bands, shuffle on bucket id, group, write.

    `partitions` is a list of file groups produced by Curator's FilePartitioningStage
    (each group is ~2 GiB by default). Partitions are distributed round-robin across
    `num_workers` actors. All `num_bands` bands are shuffled in a single pass.
    """
    num_partitions = len(partitions)
    total_num_partitions = num_output_files or num_partitions

    actors = setup_ray_ucxx_cluster(
        BulkRayShufflerActor,
        num_workers=num_workers,
        total_nparts=total_num_partitions,
        shuffle_on=["_bucket_id"],
        batchsize=1,
        output_path=output_path,
        enable_statistics=enable_statistics,
        rmm_pool_size=rmm_pool_size,
        rmm_async=rmm_async,
        pinned_memory=pinned_memory,
        pinned_initial_pool_size=pinned_initial_pool_size,
        spill_device=spill_device,
    )
    # Round-robin partition assignment so size variance is spread across actors.
    actor_partitions: list[list[list[str]]] = [[] for _ in range(num_workers)]
    for i, part in enumerate(partitions):
        actor_partitions[i % num_workers].append(part)
    print(
        f"Distributing {num_partitions} partitions across {num_workers} actors "
        f"(min/max per actor: {min(len(p) for p in actor_partitions)} / {max(len(p) for p in actor_partitions)})"
    )

    start_time = time.time()
    insert_tasks = [
        actor.lsh_read_and_insert.remote(
            actor_partitions[i],
            band_range=(0, num_bands),
            minhashes_per_band=minhashes_per_band,
            id_field=id_field,
            minhash_field=minhash_field,
        )
        for i, actor in enumerate(actors)
    ]
    ray.get(insert_tasks)
    ray.get([actor.lsh_extract_and_write.remote(id_field) for actor in actors])
    end_time = time.time()
    print(f"LSH shuffle time: {end_time - start_time} seconds")
    ray.get([actor.cleanup.remote() for actor in actors])


def dir_path(path: str) -> Path:
    """
    Validate that the given path is a directory and return a Path object.

    Parameters
    ----------
    path
        The path to check.

    Returns
    -------
    Path
        A Path object representing the directory.

    Raises
    ------
    ValueError
        If the path is not a directory.
    """
    ret = Path(path)
    if not ret.is_dir():
        raise ValueError(f"{path} path is not a directory")
    return ret


def setup_and_run(args: argparse.Namespace) -> None:
    """
    Setup and run the bulk shuffle operation.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed command line arguments.
    """
    if args.ray_address or os.environ.get("RAY_ADDRESS") is not None:
        ray.init(address="auto")  # connect to existing cluster
    else:
        ray.init(num_cpus=64, num_gpus=args.num_workers, dashboard_host="0.0.0.0")

    import json

    with Path(args.partitions_json).open() as fh:
        partitions = json.load(fh)
    lsh_bulk_ray_shuffle(
        partitions=partitions,
        output_path=args.output,
        num_bands=args.num_bands,
        minhashes_per_band=args.minhashes_per_band,
        id_field=args.id_field,
        minhash_field=args.minhash_field,
        num_workers=args.num_workers,
        num_output_files=args.n_output_files,
        enable_statistics=args.statistics,
        rmm_pool_size=args.rmm_pool_size,
        rmm_async=args.rmm_async,
        pinned_memory=args.pinned_memory,
        pinned_initial_pool_size=args.pinned_initial_pool_size,
        spill_device=args.spill_device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Bulk-synchronous Ray shuffle",
        description="Shuffle a dataset at rest (on disk) on both ends.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of workers to use.",
    )
    parser.add_argument(
        "partitions_json",
        type=str,
        metavar="PARTITIONS_JSON",
        help="Path to JSON file produced by make_partitions.py (list[list[str]]).",
    )
    parser.add_argument(
        "output",
        type=dir_path,
        metavar="OUTPUT_DIR_PATH",
        help="Output directory path.",
    )
    parser.add_argument(
        "--num-bands",
        type=int,
        default=20,
        help="Number of LSH bands to process in a single shuffle pass (bands [0, num_bands)).",
    )
    parser.add_argument(
        "--minhashes-per-band",
        type=int,
        default=13,
        help="Number of minhashes per LSH band.",
    )
    parser.add_argument(
        "--id-field",
        type=str,
        default="_curator_dedup_id",
        help="Document id column.",
    )
    parser.add_argument(
        "--minhash-field",
        type=str,
        default="_minhash_signature",
        help="Minhash list column.",
    )
    parser.add_argument(
        "--n-output-files",
        type=int,
        default=None,
        help="Number of output files. Default preserves input file count.",
    )
    parser.add_argument(
        "--rmm-pool-size",
        type=parse_bytes,
        default=format_bytes(int(rmm.mr.available_device_memory()[1] * 0.8)),
        help=(
            "The size of the RMM pool as a string with unit such as '2MiB' and '4KiB'. "
            "With --rmm-async, this is used as the async resource initial pool size "
            "and release threshold. Default to 80%% of the total device memory, "
            "which is %(default)s."
        ),
    )
    parser.add_argument(
        "--rmm-async",
        default=False,
        action="store_true",
        help="Use RMM's cudaMallocAsync-backed memory resource.",
    )
    parser.add_argument(
        "--pinned-memory",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Use pinned host memory for spilling when available.",
    )
    parser.add_argument(
        "--pinned-initial-pool-size",
        type=parse_bytes,
        default=None,
        help=(
            "Initial pinned host memory pool size as a string with unit such as "
            "'2GiB'. Default uses RapidsMPF's pinned memory resource default."
        ),
    )
    parser.add_argument(
        "--spill-device",
        type=lambda x: None if x is None else parse_bytes(x),
        default=None,
        help=(
            "Spilling device-to-host threshold as a string with unit such as '2MiB' "
            "and '4KiB'. Default is no spilling"
        ),
    )
    parser.add_argument(
        "--statistics",
        default=False,
        action="store_true",
        help="Enable statistics.",
    )
    parser.add_argument(
        "--ray-address",
        type=str,
        default=None,
        help="Connect to an existing Ray cluster.",
    )
    args = parser.parse_args()
    args.rmm_pool_size = (args.rmm_pool_size // 256) * 256  # Align to 256 bytes
    setup_and_run(args)
