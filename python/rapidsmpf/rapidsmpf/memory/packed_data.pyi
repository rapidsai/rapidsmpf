# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from typing import Self

from pylibcudf.contiguous_split import PackedColumns as CudfPackedColumns
from rmm.pylibrmm.stream import Stream

from rapidsmpf.memory.buffer_resource import BufferResource

class PackedData:
    def __init__(self) -> None: ...
    @classmethod
    def from_cudf_packed_columns(
        cls: type[Self],
        packed_columns: CudfPackedColumns,
        stream: Stream,
        br: BufferResource,
    ) -> Self: ...
