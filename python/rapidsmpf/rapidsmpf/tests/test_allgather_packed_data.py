# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from rapidsmpf.coll.allgather import AllGather
from rapidsmpf.memory.buffer_resource import BufferResource
from rapidsmpf.memory.packed_data import PackedData

if TYPE_CHECKING:
    import rmm

    from rapidsmpf.communicator.communicator import Communicator


def test_allgather_packed_data_roundtrip(
    comm: Communicator,
    device_mr: rmm.mr.CudaMemoryResource,
) -> None:
    br = BufferResource(device_mr)
    packed = PackedData.from_host_bytes(b"hello world", br)

    gather = AllGather(comm, op_id=1, br=br)
    gather.insert(0, packed)
    gather.insert_finished()

    result = gather.wait_and_extract(ordered=True)
    assert len(result) == 1
    assert result[0].to_host_bytes() == b"hello world"
