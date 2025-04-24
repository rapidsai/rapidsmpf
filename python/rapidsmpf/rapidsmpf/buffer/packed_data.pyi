# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.contiguous_split import PackedColumns as CudfPackedColumns

class PackedData:
    def __init__(self, packed_columns: CudfPackedColumns) -> None: ...
