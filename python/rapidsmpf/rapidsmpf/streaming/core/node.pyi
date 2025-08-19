# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterable

class Node:
    pass

def run_streaming_pipeline(nodes: Iterable[Node]) -> None: ...
