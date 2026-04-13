# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from rapidsmpf.config import Options
from rapidsmpf.integrations.single import (
    destroy_worker,
    get_worker_context,
    setup_worker,
)


def test_setup_single_worker_with_stats() -> None:
    """
    setup_worker calls rmpf_worker_local_setup(None, ...); when worker is None,
    the shutdown finalizer is attached to the context with name "rapidsmpf_worker_ctx".
    """
    options = Options({"single_statistics": "true", "single_print_statistics": "true"})
    setup_worker(options=options)
    assert get_worker_context() is not None
    destroy_worker()
    with pytest.raises(RuntimeError, match="Must call setup_worker first"):
        get_worker_context()
