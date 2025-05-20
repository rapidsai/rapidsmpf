# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Mapping
from typing import TypeVar

T = TypeVar("T")

class Options:
    def __init__(
        self,
        options_as_strings: Mapping[str, str],
    ) -> None: ...
    def get_or_assign(self, key: str, parser_type: type[T], default_value: T) -> T: ...

def get_environment_variables(key_regex: str = ...) -> dict[str, str]: ...
