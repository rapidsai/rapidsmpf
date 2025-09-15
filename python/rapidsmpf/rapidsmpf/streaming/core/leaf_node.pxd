# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libcpp.vector cimport vector

from rapidsmpf.streaming.core.channel cimport cpp_Message


cdef class DeferredMessages:
    cdef vector[cpp_Message] _messages
