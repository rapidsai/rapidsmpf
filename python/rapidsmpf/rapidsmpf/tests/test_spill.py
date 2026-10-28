# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest

from rapidsmpf.memory.buffer_resource import BufferResource
from rapidsmpf.memory.spill import spill_partitions, unspill_partitions
from rapidsmpf.testing import generate_packed_data, validate_packed_data

if TYPE_CHECKING:
    import rmm.mr
    from rmm.pylibrmm.stream import Stream


@pytest.mark.parametrize("num_elements", [0, 1, 10])
@pytest.mark.parametrize("num_partitions", [1, 2, 10])
def test_spill_unspill_roundtrip(
    device_mr: rmm.mr.CudaMemoryResource,
    stream: Stream,
    num_elements: int,
    num_partitions: int,
) -> None:
    br = BufferResource(device_mr)
    partitions = [
        generate_packed_data(num_elements, partition_id * 100, stream, br)
        for partition_id in range(num_partitions)
    ]

    spilled = spill_partitions(partitions, br=br)
    unspilled = unspill_partitions(spilled, br=br, allow_overbooking=False)

    assert len(unspilled) == num_partitions
    for partition_id, partition in enumerate(unspilled):
        validate_packed_data(partition, num_elements, partition_id * 100)


def test_unspill_attributes_are_recorded(
    device_mr: rmm.mr.CudaMemoryResource,
    stream: Stream,
) -> None:
    br = BufferResource(device_mr)
    spilled = spill_partitions(
        [generate_packed_data(10, 0, stream, br)],
        br=br,
    )
    br.start_reservation_event_recording()

    unspilled = unspill_partitions(
        spilled,
        br=br,
        allow_overbooking=False,
        attributes={"rapidsmpf.site": "test.unspill", "operator": 7},
    )
    events = br.drain_reservation_events()

    submitted = next(
        cast("Any", event) for event in events if event["kind"] == "request_submitted"
    )
    assert submitted["attributes"] == {
        "rapidsmpf.site": "test.unspill",
        "operator": 7,
    }
    del unspilled
    br.stop_reservation_event_recording()
