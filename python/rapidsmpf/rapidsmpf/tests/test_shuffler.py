# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from rapidsmpf.memory.buffer_resource import BufferResource
from rapidsmpf.shuffler import Shuffler
from rapidsmpf.testing import make_partition_data, validate_partition_data

if TYPE_CHECKING:
    import rmm.mr
    from rmm.pylibrmm.stream import Stream

    from rapidsmpf.communicator.communicator import Communicator


@pytest.mark.parametrize("total_num_partitions", [1, 2, 5, 10])
@pytest.mark.parametrize("total_num_rows", [1, 9, 100, 100_000])
def test_shuffler_round_trip(
    comm: Communicator,
    device_mr: rmm.mr.CudaMemoryResource,
    stream: Stream,
    total_num_partitions: int,
    total_num_rows: int,
) -> None:
    """End-to-end shuffle of a conserved, front-loaded data model."""
    br = BufferResource(device_mr)
    shuffler = Shuffler(
        comm,
        op_id=0,
        total_num_partitions=total_num_partitions,
        br=br,
    )

    # Insert every owned input region, wait, then extract and validate.
    for local_pidx in shuffler.local_partitions():
        chunks = make_partition_data(
            total_num_partitions, total_num_rows, local_pidx, stream, br
        )
        if chunks:
            shuffler.insert_chunks(chunks)
    shuffler.insert_finished()
    shuffler.wait()

    for local_pidx in shuffler.local_partitions():
        validate_partition_data(
            shuffler.extract(local_pidx),
            total_num_partitions,
            total_num_rows,
            local_pidx,
        )

    shuffler.shutdown()
