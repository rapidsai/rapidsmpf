# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Entry point shim for the rrun binary bundled in this wheel."""

from __future__ import annotations

import os
import sys
from importlib.resources import as_file, files
from typing import NoReturn


def main() -> NoReturn:
    rrun = files("librapidsmpf").joinpath("bin/rrun")
    with as_file(rrun) as binary:
        os.execv(str(binary), [str(binary), *sys.argv[1:]])
