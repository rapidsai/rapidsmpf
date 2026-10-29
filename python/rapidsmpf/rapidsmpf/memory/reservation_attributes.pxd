# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stddef cimport size_t
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map


cdef extern from "<rapidsmpf/memory/reservation_attributes.hpp>" namespace "rapidsmpf":
    cdef cppclass cpp_ReservationAttributeValue "rapidsmpf::ReservationAttributeValue":
        size_t index() noexcept
    ctypedef unordered_map[string, cpp_ReservationAttributeValue] \
        cpp_ReservationAttributeMap "rapidsmpf::ReservationAttributeMap"

    ctypedef shared_ptr[const cpp_ReservationAttributeMap] cpp_ReservationAttributes \
        "rapidsmpf::ReservationAttributes"


cdef cpp_ReservationAttributes reservation_attributes_from_mapping(object attributes) except *
