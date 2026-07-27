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

std::pair<MemoryReservation, std::size_t> ReservationAwareResourceAdaptor::reserve(
    std::size_t size, bool allow_overbooking
) {
    auto const [granted, overbooking] = get().try_reserve(size, allow_overbooking);
    // The reservation shares ownership of the adaptor's state, so it stays alive for
    // as long as any buffer allocated from the reservation needs it.
    return {
        MemoryReservation{cuda::mr::make_shared_resource<MemoryReservation::Impl>(
            static_cast<shared_base const&>(*this), granted
        )},
        overbooking
    };
}

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

}  // namespace rapidsmpf::experimental
