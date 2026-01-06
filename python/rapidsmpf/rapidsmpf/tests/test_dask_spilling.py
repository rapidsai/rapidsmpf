# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import random
import threading
import time
from collections.abc import Iterable
from typing import TYPE_CHECKING

import pytest
from dask_cudf.backends import sizeof_dispatch as sizeof
from distributed.protocol.cuda import cuda_dumps, cuda_loads

import cudf
import rmm
from pylibcudf import gpumemoryview

from rapidsmpf.integrations.dask.spilling import (
    SpillableWrapper,
    register_dask_serialize,
)
from rapidsmpf.memory.buffer import MemoryType

if TYPE_CHECKING:
    from collections.abc import Iterable

    from rmm.pylibrmm.memory_resource import DeviceMemoryResource
    from rmm.pylibrmm.stream import Stream


register_dask_serialize()


def test_spillable_wrapper(stream: Stream, device_mr: DeviceMemoryResource) -> None:
    df = cudf.DataFrame({"a": [1, 2, 3]}, dtype="int64")

    wrapper: SpillableWrapper[cudf.DataFrame] = SpillableWrapper(on_device=df)
    assert wrapper.mem_type() == MemoryType.DEVICE
    assert wrapper.approx_spillable_amount() == sizeof(df) == 24

    wrapper.spill(amount=0, stream=stream, device_mr=device_mr)
    assert wrapper.mem_type() == MemoryType.DEVICE
    assert wrapper.approx_spillable_amount() == sizeof(df) == 24

    wrapper.spill(amount=1, stream=stream, device_mr=device_mr)
    assert wrapper.mem_type() == MemoryType.HOST
    assert wrapper.approx_spillable_amount() == 0

    res = wrapper.unspill()
    assert type(res) is type(df)
    cudf.testing.assert_eq(res, df)
    assert wrapper.mem_type() == MemoryType.DEVICE
    assert wrapper.approx_spillable_amount() == sizeof(df) == 24

    # A SpillableWrapper never deletes spilled data.
    assert wrapper._on_host is not None

    # Check that we can spill again.
    wrapper.spill(amount=1, stream=stream, device_mr=device_mr)
    assert wrapper.mem_type() == MemoryType.HOST
    assert wrapper.approx_spillable_amount() == 0


@pytest.mark.parametrize(
    "memtype",
    [
        MemoryType.DEVICE,
        MemoryType.HOST,
    ],
)
def test_spillable_wrapper_dask_serialize(
    stream: Stream, device_mr: DeviceMemoryResource, memtype: MemoryType
) -> None:
    def copy_frames(
        frames: Iterable[memoryview | gpumemoryview],
    ) -> Iterable[memoryview | gpumemoryview]:
        ret = []
        for frame in frames:
            if isinstance(frame, memoryview):
                assert frame.c_contiguous
                ret.append(memoryview(bytearray(frame)))
            else:
                cai = frame.__cuda_array_interface__
                # Must be contiguous bytes.
                assert len(cai["shape"]) == 1
                assert cai["strides"] is None or cai["strides"] == (1,)
                assert cai["typestr"] == "|u1"
                nbytes = cai["shape"][0]
                ret.append(
                    gpumemoryview(rmm.DeviceBuffer(ptr=cai["data"][0], size=nbytes))
                )
        return ret

    df = cudf.DataFrame({"a": [1, 2, 3]}, dtype="int64")
    wrapper: SpillableWrapper[cudf.DataFrame] = SpillableWrapper(on_device=df)
    if memtype == MemoryType.HOST:
        wrapper.spill(100, stream=stream, device_mr=device_mr)
    header, frames = cuda_dumps(wrapper)
    res = cuda_loads(copy.deepcopy(header), copy_frames(frames))
    assert isinstance(res, SpillableWrapper)
    cudf.testing.assert_eq(res.unspill(), df)


def test_spillable_wrapper_thread_safety(
    stream: Stream, device_mr: DeviceMemoryResource
) -> None:
    """Spawn threads and have them spill/serialize wrapped objects"""

    SEED = 42
    NUM_WRAPPERS = 20
    NUM_THREADS = 10
    NUM_OPS = 40

    # Create a lot of wrapped dataframes.
    wrappers: list[SpillableWrapper[cudf.DataFrame]] = [
        SpillableWrapper(on_device=cudf.DataFrame({"a": [i]}))
        for i in range(NUM_WRAPPERS)
    ]

    # Create threads that spill/unspill at random.
    def thread1(seed: int) -> None:
        random.seed(seed)
        for _ in range(NUM_OPS):
            idx = random.randint(0, len(wrappers) - 1)
            wrappers[idx].spill(1, stream=stream, device_mr=device_mr)

            idx = random.randint(0, len(wrappers) - 1)
            assert wrappers[idx].unspill()["a"][0] == idx
            time.sleep(0)

    threads = [
        threading.Thread(target=thread1, args=(SEED + i,)) for i in range(NUM_THREADS)
    ]

    # Create threads that serialize/deserialize at random.
    def thread2(seed: int) -> None:
        random.seed(seed)
        for _ in range(NUM_OPS):
            idx = random.randint(0, len(wrappers) - 1)
            res = cuda_loads(*cuda_dumps(wrappers[idx]))
            assert isinstance(res, SpillableWrapper)
            cudf.testing.assert_eq(res.unspill(), wrappers[idx].unspill())
            time.sleep(0)

    threads += [
        threading.Thread(target=thread2, args=(SEED + i + NUM_THREADS,))
        for i in range(NUM_THREADS)
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
