# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from rmm.pylibrmm.stream import Stream

from rapidsmpf.cuda_stream import is_equal_streams


def test_equal_same_object() -> None:
    s = Stream()
    assert is_equal_streams(s, s) is True


def test_not_equal_different_streams() -> None:
    s1 = Stream()
    s2 = Stream()
    assert is_equal_streams(s1, s2) is False


@pytest.mark.parametrize("a,b", [(None, Stream()), (Stream(), None), (None, None)])
def test_rejects_none(a: Stream | None, b: Stream | None) -> None:
    with pytest.raises(TypeError):
        is_equal_streams(a, b)
