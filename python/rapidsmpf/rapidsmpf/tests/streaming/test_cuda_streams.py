# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0 All rights reserved.
from __future__ import annotations

from typing import TYPE_CHECKING

from rapidsmpf.cuda_stream import is_equal_streams

if TYPE_CHECKING:
    from rmm.pylibrmm.stream import Stream

    from rapidsmpf.streaming.core.context import Context


def test_get_stream_from_pool(context: Context, stream: Stream) -> None:
    pool_size = context.stream_pool_size()
    assert pool_size > 1

    streams = [context.get_stream_from_pool() for _ in range(pool_size)]
    # check that all streams are different from each other
    for i in range(pool_size):
        for j in range(i + 1, pool_size):
            assert not is_equal_streams(streams[i], streams[j])
        # not equal to the default stream
        assert not is_equal_streams(streams[i], stream)
