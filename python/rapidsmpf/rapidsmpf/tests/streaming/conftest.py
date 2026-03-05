# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import pytest

import rmm.mr

from rapidsmpf.communicator.single import (
    new_communicator as single_process_comm,
)
from rapidsmpf.config import Options, get_environment_variables
from rapidsmpf.memory.buffer_resource import BufferResource
from rapidsmpf.progress_thread import ProgressThread
from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor
from rapidsmpf.streaming.core.context import Context

if TYPE_CHECKING:
    from collections.abc import Generator

    from rapidsmpf.communicator.communicator import Communicator


@pytest.fixture(scope="session")
def comm() -> Communicator:
    options = Options(get_environment_variables())
    return single_process_comm(options, ProgressThread())


@pytest.fixture
def context(comm: Communicator) -> Generator[Context, None, None]:
    """
    Fixture to get a streaming context.
    """
    options = Options(get_environment_variables())
    mr = RmmResourceAdaptor(rmm.mr.CudaMemoryResource())
    br = BufferResource(mr)

    with Context(comm.logger, br, options) as ctx:
        yield ctx


@pytest.fixture(scope="session")
def py_executor() -> ThreadPoolExecutor:
    """
    Fixture to get a streaming context.
    """
    return ThreadPoolExecutor(max_workers=1)
