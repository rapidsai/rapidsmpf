# Copyright (c) 2025, NVIDIA CORPORATION.  # noqa: D100
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from __future__ import annotations

import ctypes
import os
from pathlib import Path

# Loading with RTLD_LOCAL adds the library itself to the loader's
# loaded library cache without loading any symbols into the global
# namespace. This allows libraries that express a dependency on
# this library to be loaded later and successfully satisfy this dependency
# without polluting the global symbol table with symbols from
# libwholegraph that could conflict with symbols from other DSOs.
PREFERRED_LOAD_FLAG = ctypes.RTLD_LOCAL


def _load_system_installation(soname: str) -> ctypes.CDLL:
    """
    Try to dlopen() the library indicated by ``soname``.

    Raises ``OSError`` if library cannot be loaded.

    """
    return ctypes.CDLL(soname, PREFERRED_LOAD_FLAG)


def _load_wheel_installation(soname: str) -> ctypes.CDLL | None:
    """
    Try to dlopen() the library indicated by ``soname``.

    Returns ``None`` if the library cannot be loaded.
    """
    if (lib := Path(__file__).parent / "lib64" / soname).is_file():
        return ctypes.CDLL(str(lib), PREFERRED_LOAD_FLAG)
    return None


def load_library() -> list:
    """Dynamically load librapidsmpf.so and its dependencies."""
    try:
        # these libraries must be loaded before librapidsmpf because
        # librapidsmpf references their symbols
        import libcudf
        import librmm
        import libucxx
        import rapids_logger

        librmm.load_library()
        rapids_logger.load_library()
        libcudf.load_library()
        libucxx.load_library()
    except ModuleNotFoundError:
        # 'librapidsmpf' has a runtime dependency on 'librmm' et al.. However,
        # that dependency might be satisfied by a conda package
        # (which does not have any Python modules), instead of a wheel.
        #
        # In that situation, assume that a shared object is in a place where
        # the loader can find it.
        pass

    prefer_system_installation = (
        os.getenv("RAPIDS_LIBRAPIDSMPF_PREFER_SYSTEM_LIBRARY", "false").lower()
        != "false"
    )

    libs_to_return = []
    for soname in ["librapidsmpf.so"]:
        librapidsmpf_lib = None
        if prefer_system_installation:
            # Prefer a system library if one is present to
            # avoid clobbering symbols that other packages might expect, but if no
            # other library is present use the one in the wheel.
            try:
                librapidsmpf_lib = _load_system_installation(soname)
            except OSError:
                librapidsmpf_lib = _load_wheel_installation(soname)
        else:
            # Prefer the libraries bundled in this package. If they aren't found
            # (which might be the case in builds where the library was prebuilt before
            # packaging the wheel), look for a system installation.
            try:
                librapidsmpf_lib = _load_wheel_installation(soname)
                if librapidsmpf_lib is None:
                    librapidsmpf_lib = _load_system_installation(soname)
            except OSError:
                # If none of the searches above succeed, just silently return None
                # and rely on other mechanisms (like RPATHs on other DSOs) to
                # help the loader find the library.
                pass
        if librapidsmpf_lib:
            libs_to_return.append(librapidsmpf_lib)

    # The caller almost never needs to do anything with these libraries, but no
    # harm in offering the option since these objects at least provide handles
    # to inspect where libcugraph was loaded from.

    return libs_to_return
