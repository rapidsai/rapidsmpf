/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <limits>

#include <rapidsmpf/buffer/resource.hpp>
#include <rapidsmpf/cuda_stream.hpp>
#include <rapidsmpf/error.hpp>

namespace rapidsmpf {


MemoryReservation::~MemoryReservation() noexcept {
    clear();
}

void MemoryReservation::clear() noexcept {
    if (size_ > 0) {
        br_->release(*this, size_);
    }
}

BufferResource::BufferResource(
    rmm::device_async_resource_ref device_mr,
    std::unordered_map<MemoryType, MemoryAvailable> memory_available,
    std::optional<Duration> periodic_spill_check,
    std::shared_ptr<rmm::cuda_stream_pool> stream_pool,
    std::shared_ptr<Statistics> statistics
)
    : device_mr_{device_mr},
      memory_available_{std::move(memory_available)},
      stream_pool_{std::move(stream_pool)},
      spill_manager_{this, periodic_spill_check},
      statistics_{std::move(statistics)} {
    for (MemoryType mem_type : MEMORY_TYPES) {
        // Add missing memory availability functions.
        memory_available_.try_emplace(mem_type, std::numeric_limits<std::int64_t>::max);
    }
    RAPIDSMPF_EXPECTS(stream_pool_ != nullptr, "the stream pool pointer cannot be NULL");
    RAPIDSMPF_EXPECTS(statistics_ != nullptr, "the statistics pointer cannot be NULL");
}

std::pair<MemoryReservation, std::size_t> BufferResource::reserve(
    MemoryType mem_type, std::size_t size, bool allow_overbooking
) {
    if (size == 0) {
        return {MemoryReservation(mem_type, this, 0), 0};
    }
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

MemoryReservation BufferResource::reserve_and_spill(
    MemoryType mem_type, size_t size, bool allow_overbooking
) {
    // reserve device memory with overbooking
    auto [reservation, ob] = reserve(mem_type, size, true);

    // ask the spill manager to make room for overbooking
    if (ob > 0) {
        RAPIDSMPF_EXPECTS(
            mem_type < LowestSpillType,
            "Allocating on the lowest spillable memory type resulted in overbooking",
            std::overflow_error
        );

        // TODO: spill functions should be aware of the memory type it should spill to
        auto spilled = spill_manager_.spill(ob);
        RAPIDSMPF_EXPECTS(
            allow_overbooking || spilled >= ob,
            "failed to spill enough memory (reserved: " + format_nbytes(size)
                + ", overbooking: " + format_nbytes(ob)
                + ", spilled: " + format_nbytes(spilled) + ")",
            std::overflow_error
        );
    }

    return std::move(reservation);
}

MemoryReservation BufferResource::reserve_or_fail(
    size_t size, std::optional<MemoryType> mem_type
) {
    if (mem_type) {
        auto [res, _] = reserve(*mem_type, size, false);
        RAPIDSMPF_EXPECTS(
            res.size() == size, "failed to reserve memory", std::runtime_error
        );
        return std::move(res);
    }

    // try to allocate data buffer from memory types in order [DEVICE, HOST]
    for (auto mem_type : MEMORY_TYPES) {
        auto [res, _] = reserve(mem_type, size, false);
        if (res.size() == size) {
            return std::move(res);
        }
    }
    RAPIDSMPF_FAIL("failed to reserve memory", std::runtime_error);
}

std::size_t BufferResource::release(MemoryReservation& reservation, std::size_t size) {
    std::lock_guard const lock(mutex_);
    RAPIDSMPF_EXPECTS(
        size <= reservation.size_,
        "MemoryReservation(" + format_nbytes(reservation.size_) + ") isn't big enough ("
            + format_nbytes(size) + ")",
        std::overflow_error
    );
    std::size_t& reserved = memory_reserved(reservation.mem_type_);
    RAPIDSMPF_EXPECTS(reserved >= size, "corrupted reservation stat");
    reserved -= size;
    return reservation.size_ -= size;
}

std::unique_ptr<Buffer> BufferResource::allocate(
    std::size_t size, rmm::cuda_stream_view stream, MemoryReservation& reservation
) {
    std::unique_ptr<Buffer> ret;
    switch (reservation.mem_type_) {
    case MemoryType::HOST:
        // TODO: use pinned memory, maybe use rmm::mr::pinned_memory_resource and
        // std::pmr::vector?
        ret = std::unique_ptr<Buffer>(
            new Buffer(std::make_unique<std::vector<uint8_t>>(size), stream)
        );
        break;
    case MemoryType::DEVICE:
        ret = std::unique_ptr<Buffer>(
            new Buffer(std::make_unique<rmm::device_buffer>(size, stream, device_mr_))
        );
        break;
    default:
        RAPIDSMPF_FAIL("MemoryType: unknown");
    }
    release(reservation, size);
    return ret;
}

std::unique_ptr<Buffer> BufferResource::allocate(
    rmm::cuda_stream_view stream, MemoryReservation&& reservation
) {
    return allocate(reservation.size(), stream, reservation);
}

std::unique_ptr<Buffer> BufferResource::move(
    std::unique_ptr<rmm::device_buffer> data, rmm::cuda_stream_view stream
) {
    auto upstream = data->stream();
    if (upstream.value() != stream.value()) {
        cuda_stream_join(stream, upstream);
        data->set_stream(stream);
    }
    return std::unique_ptr<Buffer>(new Buffer(std::move(data)));
}

std::unique_ptr<Buffer> BufferResource::move(
    std::unique_ptr<Buffer> buffer, MemoryReservation& reservation
) {
    if (reservation.mem_type_ != buffer->mem_type()) {
        auto ret = allocate(buffer->size, buffer->stream(), reservation);
        buffer_copy(*ret, *buffer, buffer->size);
        return ret;
    }
    return buffer;
}

std::unique_ptr<rmm::device_buffer> BufferResource::move_to_device_buffer(
    std::unique_ptr<Buffer> buffer, MemoryReservation& reservation
) {
    RAPIDSMPF_EXPECTS(
        reservation.mem_type_ == MemoryType::DEVICE,
        "the memory type of MemoryReservation doesn't match",
        std::invalid_argument
    );
    auto stream = buffer->stream();
    auto ret = move(std::move(buffer), reservation)->release_device();
    RAPIDSMPF_EXPECTS(
        ret->stream().value() == stream.value(),
        "something went wrong, the Buffer's stream and the device_buffer's stream "
        "don't match"
    );
    return ret;
}

std::unique_ptr<std::vector<uint8_t>> BufferResource::move_to_host_vector(
    std::unique_ptr<Buffer> buffer, MemoryReservation& reservation
) {
    RAPIDSMPF_EXPECTS(
        reservation.mem_type_ == MemoryType::HOST,
        "the memory type of MemoryReservation doesn't match",
        std::invalid_argument
    );
    return move(std::move(buffer), reservation)->release_host();
}

rmm::cuda_stream_pool const& BufferResource::stream_pool() const {
    return *stream_pool_;
}

SpillManager& BufferResource::spill_manager() {
    return spill_manager_;
}

std::shared_ptr<Statistics> BufferResource::statistics() {
    return statistics_;
}

}  // namespace rapidsmpf
