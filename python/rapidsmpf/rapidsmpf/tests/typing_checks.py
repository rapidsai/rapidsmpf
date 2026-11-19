# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Static type checks, run by mypy and never by pytest.

Checks that the type stubs of rapidsmpf resolve to concrete types. A package
whose types mypy cannot find becomes ``Any`` when ``ignore_missing_imports`` is
set, and code using it passes unchecked.

Two kinds of checks work for this:

- ``assert_type(expr, T)``, where ``T`` does not come from rapidsmpf, such as a
  builtin. If the package becomes ``Any``, ``expr`` is ``Any`` and the check
  fails. Comparing against a type from rapidsmpf does not work, because both
  sides become ``Any``.
- A misuse silenced with ``# type: ignore[<code>]``. If the package becomes
  ``Any``, the error disappears, and ``warn_unused_ignores`` reports the
  unused ignore. This form also checks that the types reject a misuse.

pytest does not collect this file, because its name does not start with
``test_``.
"""

from __future__ import annotations

from typing import assert_type

from rapidsmpf.config import Options, get_environment_variables


def _check_options(options: Options) -> None:
    assert_type(options.get_strings(), dict[str, str])
    assert_type(options.serialize(), bytes)
    assert_type(get_environment_variables(), dict[str, str])
    options.no_such_method()  # type: ignore[attr-defined]
