/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <limits>

#include <rapidsmp/buffer/resource.hpp>

namespace rapidsmp {


MemoryReservation::~MemoryReservation() noexcept {
    if (size_ > 0) {
        br_->release(*this, mem_type_, size_);
    }
}

BufferResource::BufferResource(
    rmm::device_async_resource_ref device_mr,
    std::unordered_map<MemoryType, MemoryAvailable> memory_available,
    std::optional<std::chrono::duration<double>> periodic_spill_check
)
    : device_mr_{device_mr},
      memory_available_{std::move(memory_available)},
      spill_manager_{this, periodic_spill_check} {
    for (MemoryType mem_type : MEMORY_TYPES) {
        // Add missing memory availability functions.
        memory_available_.try_emplace(mem_type, std::numeric_limits<std::int64_t>::max);
    }
}

std::pair<MemoryReservation, std::size_t> BufferResource::reserve(
    MemoryType mem_type, std::size_t size, bool allow_overbooking
) {
    auto const& available = memory_available(mem_type);
    std::lock_guard<std::mutex> lock(mutex_);
    std::size_t& reserved = memory_reserved(mem_type);

    // Calculate the available memory _after_ the memory has been reserved.
    std::int64_t headroom =
        available()
        - (static_cast<std::int64_t>(reserved) + static_cast<std::int64_t>(size));
    // If negative, we are overbooking.
    std::size_t overbooking =
        headroom < 0 ? static_cast<std::size_t>(std::abs(headroom)) : 0;
    if (overbooking > 0 && !allow_overbooking) {
        // Cancel the reservation, overbooking isn't allowed.
        return {MemoryReservation(mem_type, this, 0), overbooking};
    }
    // Make the reservation.
    reserved += size;
    return {MemoryReservation(mem_type, this, size), overbooking};
}

std::size_t BufferResource::release(
    MemoryReservation& reservation, MemoryType target, std::size_t size
) {
    RAPIDSMP_EXPECTS(
        reservation.mem_type_ == target,
        "the memory type of MemoryReservation doesn't match",
        std::invalid_argument
    );
    std::lock_guard const lock(mutex_);
    RAPIDSMP_EXPECTS(
        size <= reservation.size_,
        "MemoryReservation(" + format_nbytes(reservation.size_) + ") isn't big enough ("
            + format_nbytes(size) + ")",
        std::overflow_error
    );
    std::size_t& reserved = memory_reserved(target);
    RAPIDSMP_EXPECTS(reserved >= size, "corrupted reservation stat");
    reserved -= size;
    return reservation.size_ -= size;
}

std::unique_ptr<Buffer> BufferResource::allocate(
    MemoryType mem_type,
    std::size_t size,
    rmm::cuda_stream_view stream,
    MemoryReservation& reservation
) {
    std::unique_ptr<Buffer> ret;
    switch (mem_type) {
    case MemoryType::HOST:
        // TODO: use pinned memory, maybe use rmm::mr::pinned_memory_resource and
        // std::pmr::vector?
        ret = std::make_unique<Buffer>(
            Buffer{std::make_unique<std::vector<uint8_t>>(size), this}
        );
        break;
    case MemoryType::DEVICE:
        ret = std::make_unique<Buffer>(
            Buffer{std::make_unique<rmm::device_buffer>(size, stream, device_mr_), this}
        );
        break;
    default:
        RAPIDSMP_FAIL("MemoryType: unknown");
    }
    release(reservation, mem_type, size);
    return ret;
}

std::unique_ptr<Buffer> BufferResource::move(std::unique_ptr<std::vector<uint8_t>> data) {
    return std::make_unique<Buffer>(Buffer{std::move(data), this});
}

std::unique_ptr<Buffer> BufferResource::move(std::unique_ptr<rmm::device_buffer> data) {
    return std::make_unique<Buffer>(Buffer{std::move(data), this});
}

std::unique_ptr<Buffer> BufferResource::move(
    MemoryType target,
    std::unique_ptr<Buffer> buffer,
    rmm::cuda_stream_view stream,
    MemoryReservation& reservation
) {
    if (target != buffer->mem_type()) {
        auto ret = buffer->copy(target, stream);
        release(reservation, target, ret->size);
        return ret;
    }
    return buffer;
}

std::unique_ptr<rmm::device_buffer> BufferResource::move_to_device_buffer(
    std::unique_ptr<Buffer> buffer,
    rmm::cuda_stream_view stream,
    MemoryReservation& reservation
) {
    return std::move(
        move(MemoryType::DEVICE, std::move(buffer), stream, reservation)->device()
    );
}

std::unique_ptr<std::vector<uint8_t>> BufferResource::move_to_host_vector(
    std::unique_ptr<Buffer> buffer,
    rmm::cuda_stream_view stream,
    MemoryReservation& reservation
) {
    return std::move(
        move(MemoryType::HOST, std::move(buffer), stream, reservation)->host()
    );
}

std::unique_ptr<Buffer> BufferResource::copy(
    MemoryType target,
    std::unique_ptr<Buffer> const& buffer,
    rmm::cuda_stream_view stream,
    MemoryReservation& reservation
) {
    auto ret = buffer->copy(target, stream);
    release(reservation, target, ret->size);
    return ret;
}

SpillManager& BufferResource::spill_manager() {
    return spill_manager_;
}
}  // namespace rapidsmp
