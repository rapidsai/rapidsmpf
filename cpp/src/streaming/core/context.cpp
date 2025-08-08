/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/buffer/resource.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/utils.hpp>

namespace rapidsmpf::streaming {


MemoryReservation rapidsmpf::streaming::Context::reserve_and_spill(std::size_t size) {
    auto [reservation, overbooking] = br()->reserve(MemoryType::DEVICE, size, true);
    if (overbooking > 0) {
        std::size_t const spilled = br()->spill_manager().spill(overbooking);
        if (spilled < overbooking) {
            RAPIDSMPF_FAIL(
                "overbooking: " + format_nbytes(overbooking)
                    + ", need: " + format_nbytes(size),
                std::overflow_error
            );
        }
    }
    return std::move(reservation);
}

}  // namespace rapidsmpf::streaming
