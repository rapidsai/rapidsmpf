# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from rapidsmpf.integrations.dask.core import bootstrap_dask_cluster

if TYPE_CHECKING:
    import multiprocessing


def connect_dask_client_from_subprocess(
    scheduler_address: str, q: multiprocessing.Queue
) -> None:
    # This function needs to be serializable to be used in a subprocess.
    from distributed import Client

    client = Client(scheduler_address)
    bootstrap_dask_cluster(client)
    q.put(obj=True)
