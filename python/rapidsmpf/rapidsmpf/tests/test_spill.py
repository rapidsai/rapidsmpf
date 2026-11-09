# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

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


def test_unspill_reservation_context_is_recorded(
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
        reservation_context=7,
    )
    assert len(unspilled) == 1
    events, dropped_event_count = br.drain_reservation_events()
    assert dropped_event_count == 0
    assert any(event[1] == 7 and event[3] > 0 for event in events)
    assert any(event[1] == 7 and event[3] < 0 for event in events)
    assert sum(event[3] for event in events if event[1] == 7) == 0
    assert br.stop_reservation_event_recording() == ([], 0)
