# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import Awaitable, Callable

from rapidsmpf.streaming.core.channel import Channel
from rapidsmpf.streaming.core.context import Context

async def shutdown_channels(ctx: Context, *chs: Channel) -> None: ...
async def await_cpp_future(
    future: asyncio.Future[None], *, on_cancel: Callable[[], Awaitable[None]] | None
) -> None: ...
