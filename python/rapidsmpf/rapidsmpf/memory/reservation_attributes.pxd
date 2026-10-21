# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport shared_ptr


cdef extern from "<rapidsmpf/memory/reservation_attributes.hpp>" namespace "rapidsmpf":
    cdef cppclass cpp_ReservationAttributeMap "rapidsmpf::ReservationAttributeMap":
        pass

    ctypedef shared_ptr[const cpp_ReservationAttributeMap] cpp_ReservationAttributes \
        "rapidsmpf::ReservationAttributes"


cdef cpp_ReservationAttributes reservation_attributes_from_mapping(object attributes) except *
