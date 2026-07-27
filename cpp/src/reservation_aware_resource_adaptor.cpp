/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <utility>

#include <rapidsmpf/reservation_aware_resource_adaptor.hpp>

namespace rapidsmpf::experimental {

ReservationAwareResourceAdaptor::ReservationAwareResourceAdaptor(
    any_device_resource primary_mr, std::int64_t limit
)
    : shared_base(cuda::mr::make_shared_resource<Impl>(std::move(primary_mr), limit)) {}

std::int64_t ReservationAwareResourceAdaptor::limit() const noexcept {
    return get().limit();
}

void ReservationAwareResourceAdaptor::set_limit(std::int64_t limit) noexcept {
    get().set_limit(limit);
}

std::int64_t ReservationAwareResourceAdaptor::current_allocated() const noexcept {
    return get().current_allocated();
}

std::int64_t ReservationAwareResourceAdaptor::total_reserved() const noexcept {
    return get().total_reserved();
}

std::int64_t ReservationAwareResourceAdaptor::available() const noexcept {
    return get().available();
}

ScopedMemoryRecord ReservationAwareResourceAdaptor::get_main_record() const {
    return get().get_main_record();
}

rmm::device_async_resource_ref
ReservationAwareResourceAdaptor::get_upstream_resource() const noexcept {
    return rmm::device_async_resource_ref{
        const_cast<any_device_resource&>(get().get_upstream_resource())
    };
}

MemoryReservation::MemoryReservation(
    ReservationAwareResourceAdaptor const& adaptor,
    std::size_t granted,
    std::size_t overbooking
)
    : shared_base{cuda::mr::make_shared_resource<Impl>(
          adaptor, safe_cast<std::int64_t>(granted)
      )},
      overbooking_{overbooking} {}

MemoryReservation ReservationAwareResourceAdaptor::reserve(
    std::size_t size, AllowOverbooking allow_overbooking
) {
    auto const [granted, overbooking] =
        get().try_reserve(size, allow_overbooking == AllowOverbooking::YES);
    return MemoryReservation{*this, granted, overbooking};
}

}  // namespace rapidsmpf::experimental
