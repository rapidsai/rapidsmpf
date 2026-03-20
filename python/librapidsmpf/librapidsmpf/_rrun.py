# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Entry-point shim for the bundled ``rrun`` executable."""

from __future__ import annotations

import importlib.metadata
import os
import sys
from importlib.resources import as_file, files
from pathlib import Path
from typing import NoReturn


def _iter_rapids_lib_dirs() -> list[str]:
    lib_dirs: list[str] = []
    seen: set[str] = set()

    # Collect lib/ and lib64/ directories from every installed RAPIDS C++
    # wheel.  These packages register a "cmake.prefix" entry point (the RAPIDS
    # convention for C++ library wheels).  rrun is a standalone executable so
    # no Python process pre-loads the excluded shared libraries (libcudf.so,
    # librmm.so, …); prepending them to LD_LIBRARY_PATH lets the dynamic
    # linker resolve them when the new process starts.
    for dist in importlib.metadata.distributions():
        for ep in dist.entry_points:
            if ep.group != "cmake.prefix":
                continue

            try:
                pkg_dir = Path(str(files(ep.value)))
            except Exception:
                break

            for subdir in ("lib", "lib64"):
                lib_dir = pkg_dir / subdir
                lib_dir_str = str(lib_dir)
                if lib_dir.is_dir() and lib_dir_str not in seen:
                    seen.add(lib_dir_str)
                    lib_dirs.append(lib_dir_str)
            break

    return lib_dirs


def main() -> NoReturn:
    # rrun is a standalone executable. Prepend RAPIDS wheel library directories
    # to LD_LIBRARY_PATH so the dynamic linker can resolve bundled shared
    # libraries before starting the process.
    env = os.environ.copy()

    lib_dirs = _iter_rapids_lib_dirs()
    existing = env.get("LD_LIBRARY_PATH")
    if existing:
        lib_dirs.append(existing)

    if lib_dirs:
        env["LD_LIBRARY_PATH"] = ":".join(lib_dirs)

    rrun = files("librapidsmpf").joinpath("bin/rrun")
    with as_file(rrun) as binary:
        os.execvpe(str(binary), [str(binary), *sys.argv[1:]], env)

    raise AssertionError("os.execvpe returned unexpectedly")
