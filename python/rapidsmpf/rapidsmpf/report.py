# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Analyze nsys reports for postbox spilling statistics.

Usage:
    python -m rapidsmpf.report <input_file> [options]

Examples
--------
    python -m rapidsmpf.report profile.nsys-rep
    python -m rapidsmpf.report profile.sqlite
    python -m rapidsmpf.report profile.nsys-rep --force-overwrite
"""

from __future__ import annotations

from rapidsmpf._spill_report import main

if __name__ == "__main__":
    main()
