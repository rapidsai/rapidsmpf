# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Mapping

def get_environment_variables(key_regex: str = ...) -> dict[str, str]: ...

class Options:
    def __init__(
        self,
        options_as_strings: Mapping[str, str],
    ) -> None: ...
