# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def event_time_ns():
    """
    Current time on the clock used to timestamp recorded events.

    The clock is monotonic with an unspecified epoch. To convert event times to
    another time domain, such as Unix time, sample this function and the target
    clock at about the same time and add the difference to each event time.

    Returns
    -------
    int
        Nanoseconds since the steady clock's epoch.
    """
    return cpp_event_time_ns()
