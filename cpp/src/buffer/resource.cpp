/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <utility>

#include <rapidsmp/buffer/resource.hpp>

namespace rapidsmp {


MemoryReservation::~MemoryReservation() noexcept {
    if (size_ > 0) {
        br_->release(*this, mem_type_, size_);
    }
}

std::pair<MemoryReservation, std::size_t> BufferResource::reserve(
    MemoryType mem_type, std::size_t size, bool allow_overbooking
) {
    auto const& available = memory_available_.at(mem_type);
    std::lock_guard<std::mutex> lock(mutex_);
    std::size_t& reserved = memory_reserved_.at(mem_type);

    // Calculate the available memory _after_ the memory has been reserved.
    std::int64_t headroom = available() - (reserved + size);
    // If negative, we are overbooking.
    std::size_t overbooking = headroom < 0 ? -headroom : 0;

    std::cout << "reserve() - size: " << size << ", headroom: " << headroom
              << ", overbooking: " << overbooking << std::endl;

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

std::unique_ptr<Buffer> BufferResource::allocate(
    std::size_t size, rmm::cuda_stream_view stream
) {
    // First, try to reserve and allocate device memory.
    {
        auto [reservation, overbooking] = reserve(MemoryType::DEVICE, size, false);
        if (reservation.size() > 0) {
            RAPIDSMP_EXPECTS(overbooking <= 0, "got an overbooking reservation");
            std::cout << "allocate(DEVICE) - size: " << size << std::endl;
            return BufferResource::allocate(
                MemoryType::DEVICE, size, stream, reservation
            );
        }
    }
    // If that didn't work because of overbooking, we allocate host memory instead.
    auto [reservation, overbooking] = reserve(MemoryType::HOST, size, false);
    RAPIDSMP_EXPECTS(
        overbooking == 0,
        "Cannot reserve " + format_nbytes(size) + " of device or host memory",
        std::overflow_error
    );
    std::cout << "allocate(HOST) - size: " << size << std::endl;
    auto ret = BufferResource::allocate(MemoryType::HOST, size, stream, reservation);
    RAPIDSMP_EXPECTS(reservation.size() == 0, "didn't use all of the reservation");
    return ret;
}

std::unique_ptr<Buffer> BufferResource::move(
    std::unique_ptr<std::vector<uint8_t>> data, rmm::cuda_stream_view stream
) {
    return std::make_unique<Buffer>(Buffer{std::move(data), this});
}

std::unique_ptr<Buffer> BufferResource::move(
    std::unique_ptr<rmm::device_buffer> data, rmm::cuda_stream_view stream
) {
    return std::make_unique<Buffer>(Buffer{std::move(data), this});
}

std::unique_ptr<Buffer> BufferResource::move(
    MemoryType target,
    std::unique_ptr<Buffer> buffer,
    rmm::cuda_stream_view stream,
    MemoryReservation& reservation
) {
    if (target != buffer->mem_type) {
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
    if (target != buffer->mem_type) {
        release(reservation, target, ret->size);
    }
    return ret;
}
}  // namespace rapidsmp
