# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import random
import threading
import time

from dask_cudf.backends import sizeof_dispatch as sizeof

import cudf

from rapidsmp.buffer.buffer import MemoryType
from rapidsmp.integrations.dask.spilling import SpillableWrapper


def test_spillable_wrapper() -> None:
    df = cudf.DataFrame({"a": [1, 2, 3]}, dtype="int64")

    wrapper: SpillableWrapper[cudf.DataFrame] = SpillableWrapper(on_device=df)
    assert wrapper.mem_type() == MemoryType.DEVICE
    assert wrapper.approx_spillable_amount() == sizeof(df) == 24

    wrapper.spill(amount=0)
    assert wrapper.mem_type() == MemoryType.DEVICE
    assert wrapper.approx_spillable_amount() == sizeof(df) == 24

    wrapper.spill(amount=1)
    assert wrapper.mem_type() == MemoryType.HOST
    assert wrapper.approx_spillable_amount() == 0

    res = wrapper.unspill()
    assert type(res) is type(df)
    cudf.testing.assert_eq(res, df)
    assert wrapper.mem_type() == MemoryType.DEVICE
    assert wrapper.approx_spillable_amount() == sizeof(df) == 24

    # A SpillableWrapper never deletes spilled data.
    assert wrapper._on_host is not None


def test_spillable_wrapper_thread_safety() -> None:
    """Spawn threads and have them spill/unspill to wrapped objects"""

    SEED = 42
    NUM_WRAPPERS = 100
    NUM_THREADS = 10

    # Create a lot of wrapped dataframes.
    wrappers: list[SpillableWrapper[cudf.DataFrame]] = [
        SpillableWrapper(on_device=cudf.DataFrame({"a": [i]}))
        for i in range(NUM_WRAPPERS)
    ]

    # Create threads to spill/unspill at random.
    def worker() -> None:
        random.seed(SEED)
        for _ in range(len(wrappers)):
            idx = random.randint(0, len(wrappers) - 1)
            wrappers[idx].spill(1)

            idx = random.randint(0, len(wrappers) - 1)
            assert wrappers[idx].unspill()["a"][0] == idx
            time.sleep(0)

    threads = [threading.Thread(target=worker) for _ in range(NUM_THREADS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
