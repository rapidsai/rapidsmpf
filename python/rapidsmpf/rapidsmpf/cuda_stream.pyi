# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.stream import Stream

def is_equal_streams(s1: Stream, s2: Stream) -> bool: ...
