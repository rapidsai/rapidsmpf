/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <utility>

#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/memory_reservation.hpp>

namespace rapidsmpf {

MemoryReservation::~MemoryReservation() noexcept {
    clear();
}

void MemoryReservation::clear() noexcept {
    if (size_ > 0) {
        br_->release(*this, size_);
    }
}

MemoryReservation::MemoryReservation(MemoryReservation&& o)
    : MemoryReservation{
          o.mem_type_, std::exchange(o.br_, nullptr), std::exchange(o.size_, 0)
      } {}

MemoryReservation& MemoryReservation::operator=(MemoryReservation&& o) noexcept {
    clear();
    mem_type_ = o.mem_type_;
    br_ = std::exchange(o.br_, nullptr);
    size_ = std::exchange(o.size_, 0);
    return *this;
}
}  // namespace rapidsmpf
