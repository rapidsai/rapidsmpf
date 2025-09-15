# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0


cdef extern from "<rapidsmpf/communicator/ucxx.hpp>" namespace "rapidsmpf::ucxx" nogil:
    cpdef enum class ProgressMode(int):
        Blocking
        Polling
        ThreadBlocking
        ThreadPolling
