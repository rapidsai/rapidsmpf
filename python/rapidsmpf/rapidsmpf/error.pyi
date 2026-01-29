# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""RapidsMPF-specific exception classes."""

class ReservationError(MemoryError):
    pass

class OutOfMemory(MemoryError):
    pass

class BadAlloc(MemoryError):
    pass
