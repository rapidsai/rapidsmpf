# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from rapidsmpf.memory.buffer_resource import BufferResource
from rapidsmpf.memory.packed_data import PackedData
from rapidsmpf.shuffler import Shuffler

if TYPE_CHECKING:
    import rmm

    from rapidsmpf.communicator.communicator import Communicator


def test_shuffler_packed_data_roundtrip(
    comm: Communicator,
    device_mr: rmm.mr.CudaMemoryResource,
) -> None:
    br = BufferResource(device_mr)
    packed = PackedData.from_host_bytes(b"hello world", br)
    shuffler = Shuffler(comm, op_id=0, total_num_partitions=1, br=br)

    try:
        shuffler.insert_chunks({0: packed})
        shuffler.insert_finished()
        shuffler.wait()

        result = shuffler.extract(0)
        assert len(result) == 1
        assert result[0].to_host_bytes() == b"hello world"
    finally:
        shuffler.shutdown()
