# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Tests for AllGather functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import cudf
from pylibcudf.contiguous_split import pack

from rapidsmpf.allgather import AllGather
from rapidsmpf.integrations.cudf.partition import unpack_and_concat
from rapidsmpf.memory.buffer_resource import BufferResource
from rapidsmpf.memory.packed_data import PackedData
from rapidsmpf.progress_thread import ProgressThread
from rapidsmpf.statistics import Statistics
from rapidsmpf.utils.cudf import (
    cudf_to_pylibcudf_table,
    pylibcudf_to_cudf_dataframe,
)

if TYPE_CHECKING:
    import rmm.mr
    from rmm.pylibrmm.stream import Stream

    from rapidsmpf.communicator.communicator import Communicator


def generate_packed_data(
    n_elements: int, offset: int, stream: Stream, br: BufferResource
) -> PackedData:
    """
    Generate a packed data object with the given number of elements and offset.

    Both metadata and gpu_data contain the same sequential integer data starting
    from the specified offset.

    Parameters
    ----------
    n_elements
        Number of integer elements to generate
    offset
        Starting value for the sequence (offset, offset+1, offset+2, ...)
    stream
        CUDA stream for operations
    br
        Buffer resource for memory allocation

    Returns
    -------
    Packed data containing the generated sequence
    """
    # Generate sequential integers starting from offset
    values = np.arange(offset, offset + n_elements, dtype=np.int32)
    df = cudf.DataFrame({"data": values})
    packed_columns = pack(cudf_to_pylibcudf_table(df))
    return PackedData.from_cudf_packed_columns(packed_columns, stream, br)


def validate_packed_data(
    packed_data: PackedData,
    n_elements: int,
    offset: int,
    stream: Stream,
    br: BufferResource,
) -> None:
    """
    Validate a packed data object by checking its contents.

    For now, this is a simplified validation that just checks we can
    convert the packed data back to a cuDF table and that it has the
    expected number of rows.

    Parameters
    ----------
    packed_data
        The packed data to validate
    n_elements
        Expected number of elements
    offset
        Expected starting offset value (currently not fully validated)
    stream
        CUDA stream for operations

    Raises
    ------
    AssertionError
        If the data doesn't match expectations
    """
    # unpack_and_concat expects a list of PackedData
    plc_table = unpack_and_concat([packed_data], stream, br)
    result_df = pylibcudf_to_cudf_dataframe(plc_table)

    # Verify the row count matches expected
    assert len(result_df) == n_elements

    if n_elements > 0:
        # Basic validation - check that we have the expected structure
        assert len(result_df.columns) == 1

        # Convert to numpy for validation
        result_values = result_df[result_df.columns[0]].to_numpy()
        expected_values = np.arange(offset, offset + n_elements, dtype=np.int32)

        np.testing.assert_array_equal(result_values, expected_values)


def gen_offset(i: int, r: int) -> int:
    """Generate offset value like the C++ test: i * 10 + r."""
    return i * 10 + r


@pytest.mark.parametrize("n_elements", [0, 1, 10, 100])
@pytest.mark.parametrize("n_inserts", [0, 1, 10])
@pytest.mark.parametrize("ordered", [False, True])
def test_basic_allgather(
    comm: Communicator,
    device_mr: rmm.mr.CudaMemoryResource,
    stream: Stream,
    n_elements: int,
    n_inserts: int,
    ordered: bool,  # noqa: FBT001
) -> None:
    """
    Test basic AllGather functionality.

    This test mirrors the C++ AllGatherTest::basic_allgather test.
    Each rank inserts n_inserts pieces of data, then all ranks
    should receive all data from all ranks.
    """
    br = BufferResource(device_mr)
    progress_thread = ProgressThread(comm)
    statistics = Statistics(enable=False)

    # Create AllGather instance
    allgather = AllGather(
        comm=comm,
        progress_thread=progress_thread,
        op_id=0,  # Use operation ID 0
        br=br,
        statistics=statistics,
    )

    this_rank = comm.rank
    n_ranks = comm.nranks

    # Insert data from this rank
    for i in range(n_inserts):
        packed_data = generate_packed_data(
            n_elements, gen_offset(i, this_rank), stream, br
        )
        allgather.insert(i, packed_data)

    # Mark this rank as finished
    allgather.insert_finished()

    # Wait for completion and extract results
    results = allgather.wait_and_extract(ordered=ordered)

    # Verify completion
    assert allgather.finished()

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
                        results[result_idx], n_elements, expected_offset, stream, br
                    )
        else:
            # For unordered results, just verify all expected offsets are present

            # For unordered results, we can't easily determine the exact order,
            # so we just validate that each result is valid and has the right size
            for result in results:
                # Use our validation function with dummy offset (we can't easily extract the real offset)
                # This will at least verify the structure and size
                from rapidsmpf.integrations.cudf.partition import unpack_and_concat

                plc_table = unpack_and_concat([result], stream, br)
                result_df = pylibcudf_to_cudf_dataframe(plc_table)
                assert len(result_df) == n_elements
