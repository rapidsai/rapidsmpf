# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import rmm.mr

import rapidsmpf.communicator.single
import rapidsmpf.config
import rapidsmpf.integrations.core
import rapidsmpf.rmm_resource_adaptor


class Worker:
    pass


@pytest.mark.parametrize("statistics", [False, True])
def test_rmpf_worker_setup_memory_resource(*, statistics: bool) -> None:
    # setup
    upstream_mr = rmm.mr.CudaMemoryResource()
    mr = rapidsmpf.rmm_resource_adaptor.RmmResourceAdaptor(upstream_mr)
    if statistics:
        mr = rmm.mr.StatisticsResourceAdaptor(mr)
    rmm.mr.set_current_device_resource(mr)

    if statistics:
        options = rapidsmpf.config.Options({"single_statistics": "true"})
    else:
        options = rapidsmpf.config.Options()
    comm = rapidsmpf.communicator.single.new_communicator(options=options)
    # call
    worker = Worker()
    worker_context = rapidsmpf.integrations.core.rmpf_worker_setup(
        worker, "single_", comm=comm, options=options
    )

    # The global is set
    assert rmm.mr.get_current_device_resource() is mr

    assert worker_context.statistics.enabled is statistics

    if statistics:
        assert isinstance(
            worker_context.statistics.mr,
            rapidsmpf.rmm_resource_adaptor.RmmResourceAdaptor,
        )
        assert worker_context.statistics.mr is mr.get_upstream()
    else:
        assert worker_context.statistics.mr is None

    assert worker_context.br.device_mr is mr
    # Can't say much about worker_context.br.memory_available, since it just returns an int
