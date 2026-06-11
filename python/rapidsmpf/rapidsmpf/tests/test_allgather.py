# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Tests for AllGather functionality."""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING

import numpy as np
import pytest

from rapidsmpf.coll import AllGather
from rapidsmpf.memory.buffer_resource import BufferResource
from rapidsmpf.testing import generate_packed_data, validate_packed_data

if TYPE_CHECKING:
    import rmm.mr
    from rmm.pylibrmm.stream import Stream

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.memory.packed_data import PackedData


def gen_offset(i: int, r: int) -> int:
    """Generate offset value like the C++ test: i * 10 + r."""
    return i * 10 + r


@pytest.mark.parametrize("n_elements", [0, 1, 10, 100])
@pytest.mark.parametrize("n_inserts", [0, 1, 10])
@pytest.mark.parametrize("ordered", [False, True])
@pytest.mark.parametrize(
    "use_context_manager", [True, False], ids=["context", "non-context"]
)
def test_basic_allgather(
    comm: Communicator,
    device_mr: rmm.mr.CudaMemoryResource,
    stream: Stream,
    n_elements: int,
    n_inserts: int,
    ordered: bool,  # noqa: FBT001
    use_context_manager: bool,  # noqa: FBT001
) -> None:
    """
    Test basic AllGather functionality.

    This test mirrors the C++ AllGatherTest::basic_allgather test.
    Each rank inserts n_inserts pieces of data, then all ranks
    should receive all data from all ranks.
    """
    br = BufferResource(device_mr)

    # Create AllGather instance
    allgather = AllGather(
        comm=comm,
        op_id=0,
        br=br,
    )

    n_ranks = comm.nranks
    this_rank = comm.rank

    cm = allgather if use_context_manager else nullcontext(allgather)
    with cm as ag:
        for i in range(n_inserts):
            packed_data = generate_packed_data(
                n_elements, gen_offset(i, this_rank), stream, br
            )
            ag.insert(i, packed_data)
    if not use_context_manager:
        allgather.insert_finished()

    # Wait for completion and extract results
    results = allgather.wait_and_extract(ordered=ordered)

    # Check results
    if n_inserts > 0:
        expected_total = n_inserts * n_ranks
        assert len(results) == expected_total

        if ordered:
            # Results should be ordered by rank and insertion order
            # Expected pattern:
            # rank0: offset(0,0), offset(1,0), ..., offset(n_inserts-1,0)
            # rank1: offset(0,1), offset(1,1), ..., offset(n_inserts-1,1)
            # ...
            # rankN: offset(0,N), offset(1,N), ..., offset(n_inserts-1,N)

            for r in range(n_ranks):
                for i in range(n_inserts):
                    result_idx = r * n_inserts + i
                    expected_offset = gen_offset(i, r)
                    validate_packed_data(
                        results[result_idx], n_elements, expected_offset
                    )
        else:
            if n_elements == 0:
                for result in results:
                    validate_packed_data(result, 0, 0)
            else:
                expected_offsets = {
                    gen_offset(i, r) for r in range(n_ranks) for i in range(n_inserts)
                }
                actual_by_offset: dict[int, PackedData] = {
                    int(
                        np.frombuffer(result.to_host_bytes(), dtype=np.int64)[0]
                    ): result
                    for result in results
                }
                assert set(actual_by_offset) == expected_offsets
                for offset, result in actual_by_offset.items():
                    validate_packed_data(result, n_elements, offset)


def test_insert_finished_raises_in_context(
    comm: Communicator,
    device_mr: rmm.mr.CudaMemoryResource,
) -> None:
    """Test that insert_finished raises when called inside a context manager."""
    br = BufferResource(device_mr)
    ag = AllGather(comm=comm, op_id=0, br=br)
    with (
        ag,
        pytest.raises(
            ValueError, match=r"Cannot call insert_finished.*within a context"
        ),
    ):
        ag.insert_finished()
    ag.wait_and_extract(ordered=True)
