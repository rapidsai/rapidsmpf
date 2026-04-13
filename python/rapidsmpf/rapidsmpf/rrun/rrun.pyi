# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

def bind(
    gpu_id: int | None = None,
    *,
    cpu: bool = True,
    memory: bool = True,
    network: bool = True,
    verbose: bool = False,
) -> None: ...
