# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.stream cimport Stream


cpdef bint is_equal_streams(Stream s1, Stream s2)
