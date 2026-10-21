# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int64_t
from libcpp cimport bool as bool_t
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string

from collections.abc import Mapping


cdef extern from *:
    """
    namespace {
    using MutableReservationAttributes =
        std::shared_ptr<rapidsmpf::ReservationAttributeMap>;

    MutableReservationAttributes cpp_make_reservation_attributes() {
        return std::make_shared<rapidsmpf::ReservationAttributeMap>();
    }

    void cpp_add_reservation_attribute(
        MutableReservationAttributes const& attributes,
        std::string key,
        std::string value
    ) {
        attributes->insert_or_assign(std::move(key), std::move(value));
    }

    void cpp_add_reservation_attribute(
        MutableReservationAttributes const& attributes,
        std::string key,
        std::int64_t value
    ) {
        attributes->insert_or_assign(std::move(key), value);
    }

    void cpp_add_reservation_attribute(
        MutableReservationAttributes const& attributes,
        std::string key,
        bool value
    ) {
        attributes->insert_or_assign(std::move(key), value);
    }

    void cpp_add_reservation_attribute(
        MutableReservationAttributes const& attributes,
        std::string key,
        double value
    ) {
        attributes->insert_or_assign(std::move(key), value);
    }

    rapidsmpf::ReservationAttributes cpp_freeze_reservation_attributes(
        MutableReservationAttributes attributes
    ) {
        return attributes;
    }
    }  // namespace
    """
    shared_ptr[cpp_ReservationAttributeMap] cpp_make_reservation_attributes()
    void cpp_add_reservation_attribute(
        const shared_ptr[cpp_ReservationAttributeMap]&,
        string,
        string,
    )
    void cpp_add_reservation_attribute(
        const shared_ptr[cpp_ReservationAttributeMap]&,
        string,
        int64_t,
    )
    void cpp_add_reservation_attribute(
        const shared_ptr[cpp_ReservationAttributeMap]&,
        string,
        bool_t,
    )
    void cpp_add_reservation_attribute(
        const shared_ptr[cpp_ReservationAttributeMap]&,
        string,
        double,
    )
    cpp_ReservationAttributes cpp_freeze_reservation_attributes(
        shared_ptr[cpp_ReservationAttributeMap]
    )


cdef cpp_ReservationAttributes reservation_attributes_from_mapping(
    object attributes
) except *:
    cdef shared_ptr[cpp_ReservationAttributeMap] result
    cdef string cpp_key
    cdef object value

    if attributes is None:
        return cpp_ReservationAttributes()
    if not isinstance(attributes, Mapping):
        raise TypeError("attributes must be a mapping or None")

    for key, value in attributes.items():
        if not isinstance(key, str):
            raise TypeError("reservation attribute keys must be str")
        if not result:
            result = cpp_make_reservation_attributes()
        cpp_key = (<str>key).encode()
        if isinstance(value, str):
            cpp_add_reservation_attribute(result, cpp_key, (<str>value).encode())
        elif isinstance(value, bool):
            cpp_add_reservation_attribute(result, cpp_key, <bool_t>value)
        elif isinstance(value, int):
            cpp_add_reservation_attribute(result, cpp_key, <int64_t>value)
        elif isinstance(value, float):
            cpp_add_reservation_attribute(result, cpp_key, <double>value)
        else:
            raise TypeError(
                "reservation attribute values must be str, int, bool, or float"
            )
    if not result:
        return cpp_ReservationAttributes()
    return cpp_freeze_reservation_attributes(result)
