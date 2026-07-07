# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

from rapidsmpf.cuda_stream import is_equal_streams

if TYPE_CHECKING:
    from rmm.pylibrmm.stream import Stream

    from rapidsmpf.streaming.core.context import Context


def test_stream_from_pool(context: Context, stream: Stream) -> None:
    pool_size = context.br().stream_pool.get_pool_size()
    assert pool_size > 0

    streams = [context.br().stream_pool.get_stream() for _ in range(pool_size)]
    # check that all streams are different from each other
    for i in range(pool_size):
        for j in range(i + 1, pool_size):
            assert not is_equal_streams(streams[i], streams[j])
        # not equal to the default stream
        assert not is_equal_streams(streams[i], stream)
